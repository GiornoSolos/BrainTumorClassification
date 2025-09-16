import { Brain, BarChart3, Target, FileText, Database, Microscope, ArrowLeft } from 'lucide-react';
import { Card } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import Link from 'next/link';

export default function ResearchPage() {
  const performanceMetrics = [
    { metric: "Overall Accuracy", value: 94.2, color: "bg-blue-500" },
    { metric: "Glioma Detection", value: 95.1, color: "bg-green-500" },
    { metric: "Meningioma Detection", value: 94.8, color: "bg-purple-500" },
    { metric: "Normal Tissue", value: 96.3, color: "bg-emerald-500" },
    { metric: "Pituitary Detection", value: 90.7, color: "bg-orange-500" }
  ];

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-blue-50 dark:from-slate-900 dark:to-slate-800">
      <nav className="flex items-center justify-between p-6 max-w-7xl mx-auto">
        <Link href="/" className="flex items-center space-x-2 text-blue-600 hover:text-blue-700">
          <ArrowLeft className="h-5 w-5" />
          <span>Back to Home</span>
        </Link>
        <div className="flex items-center space-x-2">
          <Brain className="h-8 w-8 text-blue-400" />
          <span className="text-xl font-bold text-gray-900 dark:text-white">NeuroClassify Research</span>
        </div>
      </nav>

      <div className="max-w-7xl mx-auto px-6 py-12">
        <div className="text-center space-y-4 mb-16">
          <Badge variant="outline" className="text-blue-600 border-blue-200">
            Deep Learning Research
          </Badge>
          <h1 className="text-4xl lg:text-6xl font-bold text-gray-900 dark:text-white">
            Research & Methodology
          </h1>
          <p className="text-xl text-gray-600 dark:text-gray-300 max-w-4xl mx-auto">
            ResNet50-based Convolutional Neural Network for automated brain tumor classification 
            achieving 94.2% accuracy across four tumor categories
          </p>
        </div>

        <Card className="p-8 mb-16 bg-white dark:bg-gray-800 shadow-lg">
          <h2 className="text-3xl font-bold text-gray-900 dark:text-white mb-6">
            Research Abstract
          </h2>
          <div className="prose prose-gray dark:prose-invert max-w-none">
            <p className="text-gray-600 dark:text-gray-300 leading-relaxed mb-4">
              This research presents an automated brain tumor classification system utilizing a ResNet50-based 
              convolutional neural network for medical image analysis. The model classifies brain MRI scans 
              into four distinct categories: glioma, meningioma, normal tissue, and pituitary adenoma.
            </p>
            <p className="text-gray-600 dark:text-gray-300 leading-relaxed mb-4">
              The enhanced architecture incorporates transfer learning from ImageNet, fine-tuned specifically 
              for medical imaging applications. Training employed advanced data augmentation techniques and 
              achieved 94.2% classification accuracy with robust performance across all tumor types.
            </p>
            <p className="text-gray-600 dark:text-gray-300 leading-relaxed">
              Key contributions include successful adaptation of ResNet50 for medical image classification, 
              comprehensive evaluation metrics, development of a production-ready web interface, and 
              privacy-preserving architecture suitable for medical applications.
            </p>
          </div>
        </Card>

        <Card className="p-8 mb-16 bg-white dark:bg-gray-800 shadow-lg">
          <h2 className="text-3xl font-bold text-gray-900 dark:text-white mb-8 text-center">
            Model Performance Analysis
          </h2>
          <div className="grid md:grid-cols-2 gap-8">
            <div>
              <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-6">
                Classification Accuracy by Type
              </h3>
              <div className="space-y-4">
                {performanceMetrics.map((metric, index) => (
                  <div key={index}>
                    <div className="flex justify-between text-sm mb-1">
                      <span className="text-gray-600 dark:text-gray-400">{metric.metric}</span>
                      <span className="font-medium text-gray-900 dark:text-white">{metric.value}%</span>
                    </div>
                    <Progress value={metric.value} className="h-2" />
                  </div>
                ))}
              </div>
            </div>
            <div>
              <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-6">
                Technical Specifications
              </h3>
              <div className="space-y-4">
                <div className="flex justify-between">
                  <span className="text-gray-600 dark:text-gray-400">Architecture</span>
                  <span className="text-gray-900 dark:text-white">ResNet50 + Enhanced Classifier</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600 dark:text-gray-400">Input Size</span>
                  <span className="text-gray-900 dark:text-white">224×224×3</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600 dark:text-gray-400">Training Epochs</span>
                  <span className="text-gray-900 dark:text-white">50</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600 dark:text-gray-400">Optimizer</span>
                  <span className="text-gray-900 dark:text-white">AdamW</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600 dark:text-gray-400">Learning Rate</span>
                  <span className="text-gray-900 dark:text-white">0.0001</span>
                </div>
              </div>
            </div>
          </div>
        </Card>

        <div className="grid md:grid-cols-3 gap-8 mb-16">
          <Card className="p-6 bg-white dark:bg-gray-800 shadow-lg text-center">
            <div className="p-3 bg-blue-500/10 rounded-lg w-fit mx-auto mb-4">
              <Database className="h-8 w-8 text-blue-500" />
            </div>
            <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-3">
              Data Preprocessing
            </h3>
            <ul className="text-gray-600 dark:text-gray-300 text-sm space-y-1">
              <li>Image resizing to 224×224</li>
              <li>ImageNet normalization</li>
              <li>Data augmentation</li>
              <li>Quality validation</li>
            </ul>
          </Card>

          <Card className="p-6 bg-white dark:bg-gray-800 shadow-lg text-center">
            <div className="p-3 bg-green-500/10 rounded-lg w-fit mx-auto mb-4">
              <Microscope className="h-8 w-8 text-green-500" />
            </div>
            <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-3">
              Model Training
            </h3>
            <ul className="text-gray-600 dark:text-gray-300 text-sm space-y-1">
              <li>Transfer learning</li>
              <li>Cross-entropy loss</li>
              <li>Early stopping</li>
              <li>Learning rate scheduling</li>
            </ul>
          </Card>

          <Card className="p-6 bg-white dark:bg-gray-800 shadow-lg text-center">
            <div className="p-3 bg-purple-500/10 rounded-lg w-fit mx-auto mb-4">
              <Target className="h-8 w-8 text-purple-500" />
            </div>
            <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-3">
              Evaluation Protocol
            </h3>
            <ul className="text-gray-600 dark:text-gray-300 text-sm space-y-1">
              <li>Hold-out validation</li>
              <li>Confusion matrix analysis</li>
              <li>Performance metrics</li>
              <li>Statistical testing</li>
            </ul>
          </Card>
        </div>

        <Card className="p-8 bg-gradient-to-r from-blue-50 to-purple-50 dark:from-blue-900/20 dark:to-purple-900/20 border border-blue-200 dark:border-blue-800">
          <h2 className="text-3xl font-bold text-gray-900 dark:text-white mb-6 text-center">
            Technical Implementation
          </h2>
          <div className="grid md:grid-cols-2 gap-8">
            <div>
              <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
                Deep Learning Stack
              </h3>
              <ul className="space-y-2 text-gray-600 dark:text-gray-300">
                <li>PyTorch 2.0+ framework</li>
                <li>ResNet50 pre-trained backbone</li>
                <li>Enhanced classifier head</li>
                <li>ONNX model conversion</li>
                <li>Web deployment optimization</li>
              </ul>
            </div>
            <div>
              <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
                Web Application
              </h3>
              <ul className="space-y-2 text-gray-600 dark:text-gray-300">
                <li>Next.js 15 + TypeScript</li>
                <li>ONNX.js inference engine</li>
                <li>Vercel edge deployment</li>
                <li>Real-time processing</li>
                <li>Global CDN distribution</li>
              </ul>
            </div>
          </div>
        </Card>
      </div>
    </div>
  );
}