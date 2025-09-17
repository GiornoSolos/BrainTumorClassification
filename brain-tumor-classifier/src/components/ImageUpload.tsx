'use client';

import { useState, useCallback } from 'react';
import { Upload, X, FileImage, AlertCircle, CheckCircle, ArrowLeft } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card } from '@/components/ui/card';
import { Progress } from '@/components/ui/progress';
import { Badge } from '@/components/ui/badge';
import Image from 'next/image';
import Link from 'next/link';

interface PredictionResult {
  class: string;
  confidence: number;
  explanation: string;
  processing_time?: number;
  all_probabilities?: Record<string, number>;
  model_info?: {
    architecture: string;
    accuracy: string;
  };
}

export default function ImageUpload() {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleFileSelect = useCallback((selectedFile: File) => {
    if (!selectedFile.type.startsWith('image/')) {
      setError('Please select a valid image file');
      return;
    }

    if (selectedFile.size > 10 * 1024 * 1024) { // 10MB limit
      setError('File size must be less than 10MB');
      return;
    }

    setFile(selectedFile);
    setError(null);
    setResult(null);

    // Create preview
    const reader = new FileReader();
    reader.onload = (e) => {
      setPreview(e.target?.result as string);
    };
    reader.readAsDataURL(selectedFile);
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    const droppedFile = e.dataTransfer.files[0];
    if (droppedFile) handleFileSelect(droppedFile);
  }, [handleFileSelect]);

  const handleFileInput = (e: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = e.target.files?.[0];
    if (selectedFile) handleFileSelect(selectedFile);
  };

  const analyzeImage = async () => {
    if (!file) return;

    setLoading(true);
    setError(null);

    try {
      const formData = new FormData();
      formData.append('image', file);

      const response = await fetch('/api/predict', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.details || 'Analysis failed');
      }

      const result = await response.json();
      setResult(result);
    } catch (error) {
      console.error('Analysis error:', error);
      setError(error instanceof Error ? error.message : 'Failed to analyze image. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const resetUpload = () => {
    setFile(null);
    setPreview(null);
    setResult(null);
    setError(null);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-blue-50 dark:from-slate-900 dark:to-slate-800">
      {/* Navigation */}
      <nav className="flex items-center justify-between p-6 max-w-7xl mx-auto">
        <Link href="/" className="flex items-center space-x-2 text-blue-600 hover:text-blue-700">
          <ArrowLeft className="h-5 w-5" />
          <span>Back to Home</span>
        </Link>
        <div className="flex items-center space-x-2">
          <span className="text-xl font-bold text-gray-900 dark:text-white">NeuroClassify</span>
        </div>
      </nav>

      {/* Main Content */}
      <div className="max-w-4xl mx-auto p-6 space-y-8">
        <div className="text-center space-y-4">
          <h1 className="text-4xl font-bold text-gray-900 dark:text-white">
            Brain Tumor Classifier
          </h1>
          <p className="text-lg text-gray-600 dark:text-gray-300">
            Upload an MRI scan for instant AI-powered analysis
          </p>
        </div>

        {/* Upload Area */}
        <Card className="p-8 bg-white dark:bg-gray-800 shadow-xl">
          {!preview ? (
            <div
              onDrop={handleDrop}
              onDragOver={(e) => e.preventDefault()}
              className="border-2 border-dashed border-gray-300 dark:border-gray-600 rounded-lg p-12 text-center hover:border-blue-400 transition-colors"
            >
              <Upload className="mx-auto h-12 w-12 text-gray-400 mb-4" />
              <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-2">
                Upload MRI Scan
              </h3>
              <p className="text-gray-500 dark:text-gray-400 mb-4">
                Drag and drop an image, or click to browse
              </p>
              <input
                type="file"
                accept="image/*"
                onChange={handleFileInput}
                className="hidden"
                id="file-upload"
              />
              <Button 
                variant="outline" 
                className="cursor-pointer"
                onClick={() => document.getElementById('file-upload')?.click()}
              >
                <FileImage className="mr-2 h-4 w-4" />
                Choose File
              </Button>
              <p className="text-xs text-gray-400 mt-2">
                Supports: JPG, PNG, DICOM • Max size: 10MB
              </p>
            </div>
          ) : (
            <div className="space-y-6">
              {/* Image Preview */}
              <div className="relative">
                <Image
                  src={preview}
                  alt="MRI Preview"
                  width={400}
                  height={400}
                  className="w-full max-w-md mx-auto rounded-lg shadow-lg object-contain"
                />
                <button
                  onClick={resetUpload}
                  className="absolute top-2 right-2 p-1 bg-red-500 text-white rounded-full hover:bg-red-600"
                >
                  <X className="h-4 w-4" />
                </button>
              </div>

              {/* File Info */}
              <div className="text-center space-y-2">
                <p className="font-medium text-gray-900 dark:text-white">
                  {file?.name}
                </p>
                <p className="text-sm text-gray-500">
                  {((file?.size ?? 0) / 1024 / 1024).toFixed(2)} MB
                </p>
              </div>

              {/* Analyze Button */}
              <div className="text-center">
                <Button
                  onClick={analyzeImage}
                  disabled={loading}
                  size="lg"
                  className="bg-blue-500 hover:bg-blue-600 text-white"
                >
                  {loading ? 'Analyzing...' : 'Analyze Image'}
                </Button>
              </div>

              {/* Loading Progress */}
              {loading && (
                <div className="space-y-2">
                  <div className="flex justify-between text-sm text-gray-600 dark:text-gray-400">
                    <span>Processing image...</span>
                    <span>AI Analysis in progress</span>
                  </div>
                  <Progress value={85} className="w-full" />
                </div>
              )}
            </div>
          )}
        </Card>

        {/* Error Display */}
        {error && (
          <Card className="p-4 border-red-200 bg-red-50 dark:bg-red-900/20 border">
            <div className="flex items-center space-x-2 text-red-600 dark:text-red-400">
              <AlertCircle className="h-5 w-5" />
              <span>{error}</span>
            </div>
          </Card>
        )}

        {/* Results Display */}
        {result && (
          <Card className="p-6 space-y-6 bg-white dark:bg-gray-800 shadow-xl">
            <div className="flex items-center space-x-2">
              <CheckCircle className="h-6 w-6 text-green-500" />
              <h3 className="text-xl font-semibold text-gray-900 dark:text-white">
                Analysis Complete
              </h3>
            </div>

            <div className="grid md:grid-cols-2 gap-6">
              <div className="space-y-4">
                <div>
                  <label className="text-sm font-medium text-gray-500 dark:text-gray-400">
                    Classification
                  </label>
                  <div className="flex items-center space-x-2 mt-1">
                    <Badge
                      variant={result.class === 'notumor' ? 'default' : 'destructive'}
                      className="text-lg px-3 py-1"
                    >
                      {result.class}
                    </Badge>
                  </div>
                </div>

                <div>
                  <label className="text-sm font-medium text-gray-500 dark:text-gray-400">
                    Confidence Score
                  </label>
                  <div className="mt-1">
                    <div className="flex justify-between text-sm mb-1">
                      <span className="font-medium">{result.confidence.toFixed(1)}%</span>
                      <span className="text-gray-500">Primary Prediction</span>
                    </div>
                    <Progress value={result.confidence} className="h-2" />
                  </div>
                </div>

                {result.all_probabilities && (
                  <div>
                    <label className="text-sm font-medium text-gray-500 dark:text-gray-400 mb-2 block">
                      All Probabilities
                    </label>
                    <div className="space-y-2">
                      {Object.entries(result.all_probabilities).map(([className, probability]) => (
                        <div key={className} className="flex justify-between items-center">
                          <span className="text-sm text-gray-600 dark:text-gray-400 capitalize">{className}</span>
                          <div className="flex items-center space-x-2">
                            <div className="w-20 bg-gray-200 dark:bg-gray-700 rounded-full h-1.5">
                              <div 
                                className="bg-blue-500 h-1.5 rounded-full transition-all duration-300" 
                                style={{ width: `${probability}%` }}
                              ></div>
                            </div>
                            <span className="text-sm font-medium text-gray-900 dark:text-white w-12 text-right">
                              {probability.toFixed(1)}%
                            </span>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {result.processing_time && (
                  <div className="text-xs text-gray-500 dark:text-gray-400">
                    Analysis completed in {result.processing_time}ms
                  </div>
                )}
              </div>

              <div>
                <label className="text-sm font-medium text-gray-500 dark:text-gray-400">
                  Explanation
                </label>
                <p className="mt-1 text-gray-700 dark:text-gray-300 leading-relaxed">
                  {result.explanation}
                </p>
              </div>
            </div>

            <div className="pt-4 border-t border-gray-200 dark:border-gray-700">
              <p className="text-xs text-gray-500 dark:text-gray-400">
                This tool is for research purposes only and should not be used for medical diagnosis. 
                Please consult a qualified healthcare professional for medical advice.
              </p>
            </div>
          </Card>
        )}
      </div>
    </div>
  );
}