import { NextRequest, NextResponse } from 'next/server';

// Interface for prediction results
interface PredictionResult {
  class: string;
  confidence: number;
  explanation: string;
}

export async function POST(request: NextRequest) {
  try {
    const formData = await request.formData();
    const image = formData.get('image') as File;

    if (!image) {
      return NextResponse.json(
        { error: 'No image provided' },
        { status: 400 }
      );
    }

    // Validate image
    if (!image.type.startsWith('image/')) {
      return NextResponse.json(
        { error: 'Invalid file type. Please upload an image.' },
        { status: 400 }
      );
    }

    if (image.size > 10 * 1024 * 1024) { // 10MB limit
      return NextResponse.json(
        { error: 'File too large. Maximum size is 10MB.' },
        { status: 400 }
      );
    }

    // Convert image to buffer for processing
    const bytes = await image.arrayBuffer();
    const buffer = Buffer.from(bytes);

    // Process the image with ML model
    const prediction = await predictBrainTumor(buffer, image.name);

    return NextResponse.json(prediction);

  } catch (error) {
    console.error('Prediction error:', error);
    return NextResponse.json(
      { error: 'Failed to process image. Please try again.' },
      { status: 500 }
    );
  }
}

// Simulated prediction function - replace with your actual model
async function predictBrainTumor(imageBuffer: Buffer, fileName: string): Promise<PredictionResult> {
  // Simulate processing time (remove in production)
  await new Promise(resolve => setTimeout(resolve, 2000));

  // TODO: Replace this simulation with your actual ML model
  // Example: Load TensorFlow.js model and run inference
  
  const classes = ['No Tumor', 'Glioma', 'Meningioma', 'Pituitary'];
  const randomClass = classes[Math.floor(Math.random() * classes.length)];
  const confidence = 85 + Math.random() * 10; // 85-95% confidence

  const explanations = {
    'No Tumor': 'No abnormal tissue masses detected in this MRI scan. Brain structure appears normal with typical gray and white matter distribution. All anatomical regions show expected characteristics.',
    'Glioma': 'Irregular mass with unclear boundaries detected, showing characteristics typical of glial cell tumors. The lesion exhibits heterogeneous signal intensity and potential surrounding edema.',
    'Meningioma': 'Well-defined, round mass detected near brain membrane structures. Shows characteristics consistent with meningeal tissue growth, typically benign but requiring monitoring.',
    'Pituitary': 'Mass detected in the pituitary gland region. Shows characteristics of pituitary adenoma with typical signal patterns. May affect hormone production and requires endocrine evaluation.'
  };

  // Simulate higher confidence for "No Tumor" cases (more common)
  const adjustedConfidence = randomClass === 'No Tumor' ? 
    Math.max(confidence, 90) : confidence;

  return {
    class: randomClass,
    confidence: Math.round(adjustedConfidence * 10) / 10,
    explanation: explanations[randomClass as keyof typeof explanations]
  };
}

// Enable Edge Runtime for better performance
export const runtime = 'edge';

// Handle CORS for development
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