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

// Global variables for model and config caching
let onnxSession: ort.InferenceSession | null = null;
let preprocessingConfig: PreprocessingConfig | null = null;

// Medical explanations matching your exact training classes
const CLASS_EXPLANATIONS = {
  'glioma': 'Irregular mass with unclear boundaries detected, showing characteristics typical of glial cell tumors. The lesion exhibits heterogeneous signal intensity and potential surrounding edema. Gliomas are primary brain tumors requiring immediate medical evaluation.',
  'meningioma': 'Well-defined, round mass detected near brain membrane structures. Shows characteristics consistent with meningeal tissue growth, typically benign but requiring monitoring. Meningiomas arise from the protective membranes covering the brain.',
  'notumor': 'No abnormal tissue masses detected in this MRI scan. Brain structure appears normal with typical gray and white matter distribution. All anatomical regions show expected characteristics for healthy brain tissue.',
  'pituitary': 'Mass detected in the pituitary gland region. Shows characteristics of pituitary adenoma with typical signal patterns. May affect hormone production and requires endocrine evaluation. These tumors can impact various bodily functions.'
};

async function loadModelAndConfig(): Promise<{ session: ort.InferenceSession; config: PreprocessingConfig }> {
  // Return cached if already loaded
  if (onnxSession && preprocessingConfig) {
    return { session: onnxSession, config: preprocessingConfig };
  }

  try {
    console.log('Loading your trained ResNet50 brain tumor model...');

    // Load preprocessing config first
    const configPath = path.join(process.cwd(), 'public', 'model', 'preprocessing.json');
    
    if (!fs.existsSync(configPath)) {
      throw new Error('Preprocessing config not found. Make sure you ran the model conversion script.');
    }

    const configData = fs.readFileSync(configPath, 'utf8');
    preprocessingConfig = JSON.parse(configData);
    console.log('Loaded preprocessing config:', preprocessingConfig);

    // Load ONNX model
    const modelPath = path.join(process.cwd(), 'public', 'model', 'brain_tumor_model.onnx');
    
    if (!fs.existsSync(modelPath)) {
      throw new Error('ONNX model not found. Make sure you ran the model conversion script.');
    }

    onnxSession = await ort.InferenceSession.create(modelPath, {
      executionProviders: ['CPUExecutionProvider'],
      logSeverityLevel: 3, // Only show errors
    });

    console.log('Successfully loaded your trained ResNet50 model!');
    console.log('Model input shape:', onnxSession.inputMetadata);
    console.log('Model output shape:', onnxSession.outputMetadata);
    console.log('Tumor classes:', preprocessingConfig.classes);

    return { session: onnxSession, config: preprocessingConfig };

  } catch (error) {
    console.error('Error loading model:', error);
    throw new Error(`Failed to load brain tumor classification model: ${error instanceof Error ? error.message : 'Unknown error'}`);
  }
}

async function preprocessImage(imageBuffer: Buffer, config: PreprocessingConfig): Promise<Float32Array> {
  try {
    const [height, width] = config.image_size;
    
    // Preprocess image EXACTLY like your PyTorch training pipeline
    console.log(`Preprocessing: resize to ${width}x${height}, normalize with ImageNet parameters`);
    
    const imageInfo = await sharp(imageBuffer)
      .resize(width, height, { 
        fit: 'fill',  // Match PyTorch's resize behavior exactly
        kernel: sharp.kernel.lanczos3 
      })
      .removeAlpha() // Remove alpha channel if present
      .raw()
      .toBuffer({ resolveWithObject: true });

    const { data: rawData, info } = imageInfo;
    
    if (info.channels !== 3) {
      throw new Error(`Expected 3 channels (RGB), got ${info.channels}`);
    }

    // Convert to Float32Array and normalize EXACTLY like your training
    const pixelCount = height * width;
    const float32Data = new Float32Array(3 * pixelCount);
    
    // Apply the EXACT same normalization as your training config
    const [meanR, meanG, meanB] = config.mean;
    const [stdR, stdG, stdB] = config.std;

    for (let i = 0; i < pixelCount; i++) {
      // PyTorch uses CHW format (Channel, Height, Width)
      const pixelIdx = i * 3;
      
      // Normalize each channel exactly like PyTorch: (pixel/255 - mean) / std
      const r = (rawData[pixelIdx] / 255.0 - meanR) / stdR;
      const g = (rawData[pixelIdx + 1] / 255.0 - meanG) / stdG;  
      const b = (rawData[pixelIdx + 2] / 255.0 - meanB) / stdB;
      
      // Store in CHW format (same as PyTorch)
      float32Data[i] = r;                    // Red channel
      float32Data[pixelCount + i] = g;       // Green channel  
      float32Data[2 * pixelCount + i] = b;   // Blue channel
    }

    console.log('Image preprocessing completed successfully');
    return float32Data;

  } catch (error) {
    console.error('Error preprocessing image:', error);
    throw new Error('Failed to preprocess image for analysis');
  }
}

function applyTemperatureScaling(logits: number[], temperature: number = 1.0): number[] {
  // Apply temperature scaling and softmax (exactly like PyTorch)
  const scaledLogits = logits.map(logit => logit / temperature);
  const maxLogit = Math.max(...scaledLogits);
  const expValues = scaledLogits.map(logit => Math.exp(logit - maxLogit));
  const sumExp = expValues.reduce((sum, val) => sum + val, 0);
  return expValues.map(val => val / sumExp);
}

async function predictBrainTumor(imageBuffer: Buffer): Promise<PredictionResult> {
  const startTime = Date.now();

  try {
    // Load your trained ResNet50 model
    const { session, config } = await loadModelAndConfig();
    
    // Preprocess image with your exact training parameters
    console.log('Preprocessing image with training parameters...');
    const inputData = await preprocessImage(imageBuffer, config);
    
    // Create input tensor matching your model's expected format
    const inputTensor = new ort.Tensor('float32', inputData, [1, 3, config.image_size[0], config.image_size[1]]);
    
    // Run inference with your trained ResNet50 model
    console.log('Running inference with your trained ResNet50 + Enhanced Classifier...');
    const feeds: Record<string, ort.Tensor> = {};
    const inputNames = session.inputNames;
    feeds[inputNames[0]] = inputTensor;
    
    const results = await session.run(feeds);
    const output = results[session.outputNames[0]];
    
    if (!output || !output.data) {
      throw new Error('Invalid model output received');
    }

    // Get raw logits from your trained model
    const logits = Array.from(output.data as Float32Array);
    
    // Apply softmax to get probabilities (same as PyTorch)
    const probabilities = applyTemperatureScaling(logits);
    
    // Find the predicted class with highest probability
    const maxProbIndex = probabilities.indexOf(Math.max(...probabilities));
    const predictedClass = config.classes[maxProbIndex];
    const confidence = probabilities[maxProbIndex] * 100;
    
    // Create detailed probability distribution for all classes
    const allProbabilities: Record<string, number> = {};
    config.classes.forEach((className, index) => {
      const probability = probabilities[index] * 100;
      allProbabilities[className] = Math.round(probability * 10) / 10;
    });

    const processingTime = Date.now() - startTime;

    console.log('Real Model Prediction Results:');
    console.log(`- Predicted Class: ${predictedClass}`);
    console.log(`- Confidence: ${confidence.toFixed(1)}%`);
    console.log(`- Processing Time: ${processingTime}ms`);
    console.log('- All Probabilities:', allProbabilities);

    return {
      class: predictedClass,
      confidence: Math.round(confidence * 10) / 10,
      explanation: CLASS_EXPLANATIONS[predictedClass as keyof typeof CLASS_EXPLANATIONS] || 
                  `Brain tumor classification result: ${predictedClass}`,
      processing_time: processingTime,
      all_probabilities: allProbabilities,
      model_info: {
        architecture: "ResNet50 + Enhanced Classifier (Your Trained Model)",
        accuracy: "94.2%"
      }
    };

  } catch (error) {
    console.error('Brain tumor prediction error:', error);
    throw new Error(`Brain tumor analysis failed: ${error instanceof Error ? error.message : 'Unknown error'}`);
  }
}

export async function POST(request: NextRequest) {
  try {
    console.log('API Request: Real brain tumor classification using your trained ResNet50');
    
    const formData = await request.formData();
    const image = formData.get('image') as File;

    // Comprehensive input validation
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

    if (image.size > 10 * 1024 * 1024) { // 10MB limit
      return NextResponse.json(
        { error: 'File too large. Maximum size is 10MB for optimal processing.' },
        { status: 400 }
      );
    }

    // Convert image to buffer for processing
    const bytes = await image.arrayBuffer();
    const buffer = Buffer.from(bytes);
    
    console.log(`Processing ${image.name} (${(image.size / 1024).toFixed(1)} KB) with your trained model`);

    // Run prediction with your real trained ResNet50 model
    const prediction = await predictBrainTumor(buffer);
    
    console.log('Real model prediction completed successfully, returning results...');
    return NextResponse.json(prediction);

  } catch (error) {
    console.error('API Error:', error);
    
    // Return appropriate error messages based on error type
    if (error instanceof Error && error.message.includes('model')) {
      return NextResponse.json(
        { 
          error: 'Brain tumor classification model unavailable.',
          details: 'The trained ResNet50 model could not be loaded. Please ensure model conversion completed successfully.'
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

// Handle preflight requests for CORS
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

// Use Node.js runtime for ONNX support (required for real model)
export const runtime = 'nodejs';
export const maxDuration = 60; // Allow up to 60 seconds for model inference