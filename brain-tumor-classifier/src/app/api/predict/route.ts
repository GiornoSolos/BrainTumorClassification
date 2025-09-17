import { NextRequest, NextResponse } from 'next/server';
import * as ort from 'onnxruntime-node';
import sharp from 'sharp';

interface PredictionResult {
  class: string;
  confidence: number;
  explanation: string;
  processing_time: number;
  all_probabilities: Record<string, number>;
  model_info: {
    architecture: string;
    accuracy: string;
  };
}

interface PreprocessingConfig {
  image_size: [number, number];
  mean: [number, number, number];
  std: [number, number, number];
  classes: string[];
}

// External model URL
const MODEL_URL = "https://github.com/GiornoSolos/BrainTumorClassification/releases/download/v1.0.0/brain_tumor_model.onnx";

const PREPROCESSING_CONFIG: PreprocessingConfig = {
  image_size: [224, 224],
  mean: [0.485, 0.456, 0.406],
  std: [0.229, 0.224, 0.225],
  classes: ['glioma', 'meningioma', 'notumor', 'pituitary']
};

let onnxSession: ort.InferenceSession | null = null;

const CLASS_EXPLANATIONS = {
  'glioma': 'Irregular mass with unclear boundaries detected, showing characteristics typical of glial cell tumors. The lesion exhibits heterogeneous signal intensity and potential surrounding edema. Gliomas are primary brain tumors requiring immediate medical evaluation.',
  'meningioma': 'Well-defined, round mass detected near brain membrane structures. Shows characteristics consistent with meningeal tissue growth, typically benign but requiring monitoring. Meningiomas arise from the protective membranes covering the brain.',
  'notumor': 'No abnormal tissue masses detected in this MRI scan. Brain structure appears normal with typical gray and white matter distribution. All anatomical regions show expected characteristics for healthy brain tissue.',
  'pituitary': 'Mass detected in the pituitary gland region. Shows characteristics of pituitary adenoma with typical signal patterns. May affect hormone production and requires endocrine evaluation. These tumors can impact various bodily functions.'
};

async function loadModelAndConfig(): Promise<{ session: ort.InferenceSession; config: PreprocessingConfig }> {
  if (onnxSession) {
    return { session: onnxSession, config: PREPROCESSING_CONFIG };
  }

  try {
    console.log('Loading ONNX model from external URL with onnxruntime-node');

    // Download model with memory optimization
    const response = await fetch(MODEL_URL);
    if (!response.ok) {
      throw new Error(`Failed to download model: ${response.status}`);
    }
    
    const modelBuffer = await response.arrayBuffer();
    console.log(`Model downloaded: ${(modelBuffer.byteLength / 1024 / 1024).toFixed(1)} MB`);

    // Create session with onnxruntime-node optimized for Vercel
    onnxSession = await ort.InferenceSession.create(modelBuffer, {
      executionProviders: ['cpu'],
      enableCpuMemArena: false, // Disable memory arena to reduce memory usage
      enableMemPattern: false,  // Disable memory pattern optimization to save memory
      executionMode: 'sequential', // Use sequential execution to save memory
      graphOptimizationLevel: 'basic', // Use basic optimization to save memory
    });

    console.log('ONNX session created successfully with onnxruntime-node');
    
    // Force garbage collection if available (helps with memory management)
    if (global.gc) {
      global.gc();
    }
    
    return { session: onnxSession, config: PREPROCESSING_CONFIG };

  } catch (error) {
    console.error('Model loading error:', error);
    throw new Error(`Failed to load model: ${error instanceof Error ? error.message : 'Unknown error'}`);
  }
}

async function preprocessImage(imageBuffer: Buffer, config: PreprocessingConfig): Promise<Float32Array> {
  const [height, width] = config.image_size;
  
  try {
    const imageInfo = await sharp(imageBuffer)
      .resize(width, height, { fit: 'fill', kernel: sharp.kernel.lanczos3 })
      .removeAlpha()
      .raw()
      .toBuffer({ resolveWithObject: true });

    const { data: rawData, info } = imageInfo;
    
    if (info.channels !== 3) {
      throw new Error(`Expected 3 channels (RGB), got ${info.channels}`);
    }

    const pixelCount = height * width;
    const float32Data = new Float32Array(3 * pixelCount);
    const [meanR, meanG, meanB] = config.mean;
    const [stdR, stdG, stdB] = config.std;

    // Optimized preprocessing loop
    for (let i = 0; i < pixelCount; i++) {
      const pixelIdx = i * 3;
      const r = (rawData[pixelIdx] / 255.0 - meanR) / stdR;
      const g = (rawData[pixelIdx + 1] / 255.0 - meanG) / stdG;  
      const b = (rawData[pixelIdx + 2] / 255.0 - meanB) / stdB;
      
      float32Data[i] = r;
      float32Data[pixelCount + i] = g;
      float32Data[2 * pixelCount + i] = b;
    }

    return float32Data;
  } catch (error) {
    console.error('Image preprocessing error:', error);
    throw new Error('Failed to preprocess image');
  }
}

function applyTemperatureScaling(logits: number[], temperature: number = 1.0): number[] {
  const scaledLogits = logits.map(logit => logit / temperature);
  const maxLogit = Math.max(...scaledLogits);
  const expValues = scaledLogits.map(logit => Math.exp(logit - maxLogit));
  const sumExp = expValues.reduce((sum, val) => sum + val, 0);
  return expValues.map(val => val / sumExp);
}

async function predictBrainTumor(imageBuffer: Buffer): Promise<PredictionResult> {
  const startTime = Date.now();

  try {
    const { session, config } = await loadModelAndConfig();
    const inputData = await preprocessImage(imageBuffer, config);
    
    // Create tensor with proper dimensions
    const inputTensor = new ort.Tensor('float32', inputData, [1, 3, config.image_size[0], config.image_size[1]]);
    const feeds = { [session.inputNames[0]]: inputTensor };
    
    // Run inference
    const results = await session.run(feeds);
    const output = results[session.outputNames[0]];
    
    if (!output || !output.data) {
      throw new Error('Invalid model output received');
    }

    const logits = Array.from(output.data as Float32Array);
    const probabilities = applyTemperatureScaling(logits);
    
    const maxProbIndex = probabilities.indexOf(Math.max(...probabilities));
    const predictedClass = config.classes[maxProbIndex];
    const confidence = probabilities[maxProbIndex] * 100;
    
    const allProbabilities: Record<string, number> = {};
    config.classes.forEach((className, index) => {
      allProbabilities[className] = Math.round(probabilities[index] * 100 * 10) / 10;
    });

    const processingTime = Date.now() - startTime;

    // Clean up tensors to free memory
    inputTensor.dispose?.();
    output.dispose?.();

    return {
      class: predictedClass,
      confidence: Math.round(confidence * 10) / 10,
      explanation: CLASS_EXPLANATIONS[predictedClass as keyof typeof CLASS_EXPLANATIONS] || 
                  `Brain tumor classification result: ${predictedClass}`,
      processing_time: processingTime,
      all_probabilities: allProbabilities,
      model_info: {
        architecture: "ResNet50 + Enhanced Classifier",
        accuracy: "94.2%"
      }
    };

  } catch (error) {
    console.error('Prediction error:', error);
    throw new Error(`Prediction failed: ${error instanceof Error ? error.message : 'Unknown error'}`);
  }
}

export async function POST(request: NextRequest) {
  try {
    const formData = await request.formData();
    const image = formData.get('image') as File;

    if (!image) {
      return NextResponse.json({ error: 'No image provided' }, { status: 400 });
    }

    if (!image.type.startsWith('image/')) {
      return NextResponse.json({ error: 'Invalid file type' }, { status: 400 });
    }

    if (image.size > 10 * 1024 * 1024) {
      return NextResponse.json({ error: 'File too large' }, { status: 400 });
    }

    const bytes = await image.arrayBuffer();
    const buffer = Buffer.from(bytes);
    
    const prediction = await predictBrainTumor(buffer);
    
    // Force garbage collection if available
    if (global.gc) {
      global.gc();
    }
    
    return NextResponse.json(prediction);

  } catch (error) {
    console.error('API error:', error);
    return NextResponse.json(
      { error: 'Analysis failed', details: error instanceof Error ? error.message : 'Unknown error' },
      { status: 500 }
    );
  }
}

export async function OPTIONS() {
  return new NextResponse(null, {
    status: 200,
    headers: {
      'Access-Control-Allow-Origin': '*',
      'Access-Control-Allow-Methods': 'POST, OPTIONS',
      'Access-Control-Allow-Headers': 'Content-Type',
    },
  });
}

export const runtime = 'nodejs';
export const maxDuration = 60;