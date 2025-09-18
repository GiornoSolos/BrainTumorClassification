import { NextRequest, NextResponse } from 'next/server';
import { HfInference } from '@huggingface/inference';

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

interface HuggingFaceResult {
  label: string;
  score: number;
}

// Initialize Hugging Face client
const hf = new HfInference(process.env.HUGGINGFACE_API_TOKEN);

// Your model ID from Hugging Face Hub
const MODEL_ID = process.env.HUGGINGFACE_MODEL_ID || "yourusername/brain-tumor-classifier";

const CLASS_EXPLANATIONS = {
  'glioma': 'Irregular mass with unclear boundaries detected, showing characteristics typical of glial cell tumors. The lesion exhibits heterogeneous signal intensity and potential surrounding edema. Gliomas are primary brain tumors requiring immediate medical evaluation.',
  'meningioma': 'Well-defined, round mass detected near brain membrane structures. Shows characteristics consistent with meningeal tissue growth, typically benign but requiring monitoring. Meningiomas arise from the protective membranes covering the brain.',
  'notumor': 'No abnormal tissue masses detected in this MRI scan. Brain structure appears normal with typical gray and white matter distribution. All anatomical regions show expected characteristics for healthy brain tissue.',
  'pituitary': 'Mass detected in the pituitary gland region. Shows characteristics of pituitary adenoma with typical signal patterns. May affect hormone production and requires endocrine evaluation. These tumors can impact various bodily functions.'
};

// Map common label variations to our standard names
const LABEL_MAPPING: Record<string, string> = {
  'LABEL_0': 'glioma',
  'LABEL_1': 'meningioma', 
  'LABEL_2': 'notumor',
  'LABEL_3': 'pituitary',
  '0': 'glioma',
  '1': 'meningioma',
  '2': 'notumor', 
  '3': 'pituitary'
};

async function classifyBrainTumor(imageBuffer: Buffer): Promise<PredictionResult> {
  const startTime = Date.now();

  try {
    console.log('Sending image to Hugging Face for classification...');
    console.log('Model ID:', MODEL_ID);
    console.log('Image buffer size:', imageBuffer.length);
    
    let result: HuggingFaceResult[];
    
    try {
      // Method 1: Try Hugging Face client with proper format
      console.log('Trying Hugging Face client...');
      const uint8Array = new Uint8Array(imageBuffer);
      const imageBlob = new Blob([uint8Array], { type: 'image/jpeg' });
      
      result = await hf.imageClassification({
        data: imageBlob,
        model: MODEL_ID
      });
    } catch (clientError) {
      console.log('Hugging Face client failed, trying direct fetch...');
      
      // Method 2: Direct fetch to Hugging Face API
      const uint8Array = new Uint8Array(imageBuffer);
      const response = await fetch(
        `https://api-inference.huggingface.co/models/${MODEL_ID}`,
        {
          method: 'POST',
          headers: {
            'Authorization': `Bearer ${process.env.HUGGINGFACE_API_TOKEN}`,
            'Content-Type': 'application/octet-stream',
          },
          body: uint8Array,
        }
      );

      if (!response.ok) {
        const errorText = await response.text();
        console.error('Hugging Face API error:', response.status, errorText);
        
        if (response.status === 404) {
          throw new Error('Model not found. Please check your model ID.');
        } else if (response.status === 401) {
          throw new Error('Authentication failed. Please check your API token.');
        } else if (response.status === 429) {
          throw new Error('Rate limit exceeded. Please try again later.');
        } else if (response.status === 503) {
          throw new Error('Model is loading. Please try again in a few moments.');
        } else {
          throw new Error(`API request failed: ${response.status} ${errorText}`);
        }
      }

      result = await response.json();
    }

    console.log('Hugging Face result:', result);

    if (!result || result.length === 0) {
      throw new Error('No classification results received from Hugging Face');
    }

    // Process results from Hugging Face
    const topPrediction = result[0];
    let predictedClass = topPrediction.label.toLowerCase();
    
    // Map labels if needed
    if (LABEL_MAPPING[topPrediction.label]) {
      predictedClass = LABEL_MAPPING[topPrediction.label];
    }

    const confidence = topPrediction.score * 100;

    // Create all probabilities object
    const allProbabilities: Record<string, number> = {
      'glioma': 0,
      'meningioma': 0,
      'notumor': 0,
      'pituitary': 0
    };

    // Fill in probabilities from Hugging Face results
    result.forEach((prediction: HuggingFaceResult) => {
      let className = prediction.label.toLowerCase();
      if (LABEL_MAPPING[prediction.label]) {
        className = LABEL_MAPPING[prediction.label];
      }
      
      if (allProbabilities.hasOwnProperty(className)) {
        allProbabilities[className] = Math.round(prediction.score * 100 * 10) / 10;
      }
    });

    const processingTime = Date.now() - startTime;

    return {
      class: predictedClass,
      confidence: Math.round(confidence * 10) / 10,
      explanation: CLASS_EXPLANATIONS[predictedClass as keyof typeof CLASS_EXPLANATIONS] || 
                  `Brain tumor classification result: ${predictedClass}`,
      processing_time: processingTime,
      all_probabilities: allProbabilities,
      model_info: {
        architecture: "ResNet50 + Enhanced Classifier (Hugging Face)",
        accuracy: "94.2%"
      }
    };

  } catch (error) {
    console.error('Hugging Face classification error:', error);
    
    // Handle common Hugging Face errors
    if (error instanceof Error) {
      if (error.message.includes('Model not found') || error.message.includes('404')) {
        throw new Error('Brain tumor classification model not found. Please check model deployment.');
      } else if (error.message.includes('rate limit') || error.message.includes('429')) {
        throw new Error('Service temporarily unavailable due to high demand. Please try again in a moment.');
      } else if (error.message.includes('authentication') || error.message.includes('401')) {
        throw new Error('Model authentication failed. Please check configuration.');
      } else if (error.message.includes('loading')) {
        throw new Error('Model is currently loading. Please try again in a few moments.');
      } else if (error.message.includes('timeout')) {
        throw new Error('Request timed out. Please try again.');
      }
    }
    
    throw new Error(`Classification failed: ${error instanceof Error ? error.message : 'Unknown error'}`);
  }
}

async function getDemoResult(request: NextRequest) {
  const formData = await request.formData();
  const image = formData.get('image') as File;

  if (!image) {
    return NextResponse.json({ error: 'No image provided' }, { status: 400 });
  }

  // Simulate realistic processing time
  await new Promise(resolve => setTimeout(resolve, 1800));

  // Array of realistic demo results
  const demoResults = [
    {
      class: 'glioma',
      confidence: 94.8,
      explanation: 'Irregular mass with unclear boundaries detected, showing characteristics typical of glial cell tumors. The lesion exhibits heterogeneous signal intensity and potential surrounding edema. Gliomas are primary brain tumors requiring immediate medical evaluation.',
      processing_time: 1847,
      all_probabilities: {
        'glioma': 94.8,
        'meningioma': 3.2,
        'notumor': 1.5,
        'pituitary': 0.5
      },
      model_info: {
        architecture: "ResNet50 + Enhanced Classifier (Demo)",
        accuracy: "94.2%"
      }
    },
    {
      class: 'meningioma',
      confidence: 96.3,
      explanation: 'Well-defined, round mass detected near brain membrane structures. Shows characteristics consistent with meningeal tissue growth, typically benign but requiring monitoring. Meningiomas arise from the protective membranes covering the brain.',
      processing_time: 1623,
      all_probabilities: {
        'glioma': 2.1,
        'meningioma': 96.3,
        'notumor': 1.0,
        'pituitary': 0.6
      },
      model_info: {
        architecture: "ResNet50 + Enhanced Classifier (Demo)",
        accuracy: "94.2%"
      }
    },
    {
      class: 'notumor',
      confidence: 97.2,
      explanation: 'No abnormal tissue masses detected in this MRI scan. Brain structure appears normal with typical gray and white matter distribution. All anatomical regions show expected characteristics for healthy brain tissue.',
      processing_time: 1456,
      all_probabilities: {
        'glioma': 1.2,
        'meningioma': 0.8,
        'notumor': 97.2,
        'pituitary': 0.8
      },
      model_info: {
        architecture: "ResNet50 + Enhanced Classifier (Demo)",
        accuracy: "94.2%"
      }
    },
    {
      class: 'pituitary',
      confidence: 92.7,
      explanation: 'Mass detected in the pituitary gland region. Shows characteristics of pituitary adenoma with typical signal patterns. May affect hormone production and requires endocrine evaluation. These tumors can impact various bodily functions.',
      processing_time: 1789,
      all_probabilities: {
        'glioma': 3.1,
        'meningioma': 2.4,
        'notumor': 1.8,
        'pituitary': 92.7
      },
      model_info: {
        architecture: "ResNet50 + Enhanced Classifier (Demo)",
        accuracy: "94.2%"
      }
    }
  ];

  // Return different result each time for variety
  const randomIndex = Math.floor(Math.random() * demoResults.length);
  return NextResponse.json(demoResults[randomIndex]);
}

export async function POST(request: NextRequest) {
  try {
    // Demo mode for presentations
    if (process.env.DEMO_MODE === 'true') {
      return getDemoResult(request);
    }

    // Check for required environment variables (original code)
    if (!process.env.HUGGINGFACE_API_TOKEN) {
      console.error('HUGGINGFACE_API_TOKEN not found in environment variables');
      return NextResponse.json(
        { error: 'Hugging Face API token not configured' }, 
        { status: 500 }
      );
    }

    // ... rest of your existing Hugging Face code stays the same
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
    
    console.log('Processing image:', image.name, 'Size:', image.size, 'Type:', image.type);
    
    const prediction = await classifyBrainTumor(buffer);
    
    return NextResponse.json(prediction);

  } catch (error) {
    console.error('API error:', error);
    return NextResponse.json(
      { 
        error: 'Analysis failed', 
        details: error instanceof Error ? error.message : 'Unknown error' 
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

export const maxDuration = 30;