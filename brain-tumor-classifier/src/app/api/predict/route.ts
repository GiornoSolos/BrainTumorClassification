import { NextRequest, NextResponse } from 'next/server';
import * as ort from 'onnxruntime-node';
import sharp from 'sharp';
import path from 'path';
import fs from 'fs';

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

// Preprocessing configuration - update these values to match training parameters
const PREPROCESSING_CONFIG: PreprocessingConfig = {
  image_size: [224, 224],
  mean: [0.485, 0.456, 0.406],
  std: [0.229, 0.224, 0.225],
  classes: ['glioma', 'meningioma', 'notumor', 'pituitary']
};

// Global variable for ONNX session caching
let onnxSession: ort.InferenceSession | null = null;

// Class-specific medical explanations
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
    console.log('Loading ONNX model for brain tumor classification');

    const config = PREPROCESSING_CONFIG;
    console.log('Using embedded preprocessing configuration:', config);

    // Model file path options for different deployment environments
    const possiblePaths = [
      path.join(process.cwd(), 'public', 'model', 'brain_tumor_model.onnx'),
      path.join(process.cwd(), 'model', 'brain_tumor_model.onnx'),
      './public/model/brain_tumor_model.onnx',
      './model/brain_tumor_model.onnx'
    ];

    let modelPath: string | null = null;
    
    for (const testPath of possiblePaths) {
      console.log(`Checking model path: ${testPath}`);
      if (fs.existsSync(testPath)) {
        modelPath = testPath;
        console.log(`Model located at: ${modelPath}`);
        break;
      }
    }

    if (!modelPath) {
      // Debug file system structure
      console.log('Available files in process.cwd():', fs.readdirSync(process.cwd()));
      if (fs.existsSync(path.join(process.cwd(), 'public'))) {
        console.log('Files in public:', fs.readdirSync(path.join(process.cwd(), 'public')));
        if (fs.existsSync(path.join(process.cwd(), 'public', 'model'))) {
          console.log('Files in public/model:', fs.readdirSync(path.join(process.cwd(), 'public', 'model')));
        }
      }
      throw new Error('ONNX model not found in expected deployment locations');
    }

    onnxSession = await ort.InferenceSession.create(modelPath, {
      executionProviders: ['CPUExecutionProvider'],
      logSeverityLevel: 3,
    });

    console.log('ONNX session created successfully');
    console.log('Model input metadata:', onnxSession.inputMetadata);
    console.log('Model output metadata:', onnxSession.outputMetadata);
    console.log('Classification classes:', config.classes);

    return { session: onnxSession, config };

  } catch (error) {
    console.error('Model loading error:', error);
    throw new Error(`Failed to load brain tumor classification model: ${error instanceof Error ? error.message : 'Unknown error'}`);
  }
}

async function preprocessImage(imageBuffer: Buffer, config: PreprocessingConfig): Promise<Float32Array> {
  try {
    const [height, width] = config.image_size;
    
    console.log(`Image preprocessing: resize to ${width}x${height}, normalize with specified parameters`);
    
    const imageInfo = await sharp(imageBuffer)
      .resize(width, height, { 
        fit: 'fill',
        kernel: sharp.kernel.lanczos3 
      })
      .removeAlpha()
      .raw()
      .toBuffer({ resolveWithObject: true });

    const { data: rawData, info } = imageInfo;
    
    if (info.channels !== 3) {
      throw new Error(`Expected 3 channels (RGB), got ${info.channels}`);
    }

    // Convert to Float32Array and apply normalization
    const pixelCount = height * width;
    const float32Data = new Float32Array(3 * pixelCount);
    
    const [meanR, meanG, meanB] = config.mean;
    const [stdR, stdG, stdB] = config.std;

    for (let i = 0; i < pixelCount; i++) {
      // PyTorch CHW format (Channel, Height, Width)
      const pixelIdx = i * 3;
      
      // Apply normalization: (pixel/255 - mean) / std
      const r = (rawData[pixelIdx] / 255.0 - meanR) / stdR;
      const g = (rawData[pixelIdx + 1] / 255.0 - meanG) / stdG;  
      const b = (rawData[pixelIdx + 2] / 255.0 - meanB) / stdB;
      
      // Store in CHW format
      float32Data[i] = r;
      float32Data[pixelCount + i] = g;
      float32Data[2 * pixelCount + i] = b;
    }

    console.log('Image preprocessing completed');
    return float32Data;

  } catch (error) {
    console.error('Image preprocessing error:', error);
    throw new Error('Failed to preprocess image for analysis');
  }
}

function applyTemperatureScaling(logits: number[], temperature: number = 1.0): number[] {
  // Apply temperature scaling and softmax
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
    
    console.log('Beginning image preprocessing');
    const inputData = await preprocessImage(imageBuffer, config);
    
    // Create input tensor
    const inputTensor = new ort.Tensor('float32', inputData, [1, 3, config.image_size[0], config.image_size[1]]);
    
    console.log('Running model inference');
    const feeds: Record<string, ort.Tensor> = {};
    const inputNames = session.inputNames;
    feeds[inputNames[0]] = inputTensor;
    
    const results = await session.run(feeds);
    const output = results[session.outputNames[0]];
    
    if (!output || !output.data) {
      throw new Error('Invalid model output received');
    }

    // Process model output
    const logits = Array.from(output.data as Float32Array);
    const probabilities = applyTemperatureScaling(logits);
    
    const maxProbIndex = probabilities.indexOf(Math.max(...probabilities));
    const predictedClass = config.classes[maxProbIndex];
    const confidence = probabilities[maxProbIndex] * 100;
    
    // Generate probability distribution for all classes
    const allProbabilities: Record<string, number> = {};
    config.classes.forEach((className, index) => {
      const probability = probabilities[index] * 100;
      allProbabilities[className] = Math.round(probability * 10) / 10;
    });

    const processingTime = Date.now() - startTime;

    console.log('Prediction results:');
    console.log(`Predicted class: ${predictedClass}`);
    console.log(`Confidence: ${confidence.toFixed(1)}%`);
    console.log(`Processing time: ${processingTime}ms`);
    console.log('All probabilities:', allProbabilities);

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
    throw new Error(`Brain tumor analysis failed: ${error instanceof Error ? error.message : 'Unknown error'}`);
  }
}

export async function POST(request: NextRequest) {
  try {
    console.log('Brain tumor classification API request received');
    
    const formData = await request.formData();
    const image = formData.get('image') as File;

    // Input validation
    if (!image) {
      return NextResponse.json(
        { error: 'No image provided. Please upload an MRI scan for analysis.' },
        { status: 400 }
      );
    }

    if (!image.type.startsWith('image/')) {
      return NextResponse.json(
        { error: 'Invalid file type. Please upload an image file (JPG, PNG, DICOM, etc.).' },
        { status: 400 }
      );
    }

    if (image.size > 10 * 1024 * 1024) {
      return NextResponse.json(
        { error: 'File too large. Maximum size is 10MB for optimal processing.' },
        { status: 400 }
      );
    }

    // Convert image to buffer
    const bytes = await image.arrayBuffer();
    const buffer = Buffer.from(bytes);
    
    console.log(`Processing image: ${image.name} (${(image.size / 1024).toFixed(1)} KB)`);

    // Run model prediction
    const prediction = await predictBrainTumor(buffer);
    
    console.log('Model prediction completed successfully');
    return NextResponse.json(prediction);

  } catch (error) {
    console.error('API error:', error);
    
    // Return appropriate error responses
    if (error instanceof Error && error.message.includes('model')) {
      return NextResponse.json(
        { 
          error: 'Brain tumor classification model unavailable.',
          details: 'The ResNet50 model could not be loaded. Please ensure model files are included in deployment.'
        },
        { status: 503 }
      );
    }

    return NextResponse.json(
      { 
        error: 'Failed to analyze brain scan. Please try again.',
        details: error instanceof Error ? error.message : 'Unknown processing error occurred'
      },
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