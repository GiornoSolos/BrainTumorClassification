import { ArrowRight, Brain, Shield, Zap } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card } from '@/components/ui/card';
import Link from 'next/link';

export default function Hero() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-blue-900 to-slate-900">
      {/* Navigation */}
      <nav className="flex items-center justify-between p-6 max-w-7xl mx-auto">
        <div className="flex items-center space-x-2">
          <Brain className="h-8 w-8 text-blue-400" />
          <span className="text-xl font-bold text-white">NeuroClassify</span>
        </div>
        <div className="hidden md:flex space-x-8">
          <Link href="/features" className="text-gray-300 hover:text-white transition-colors">Features</Link>
          <Link href="/about" className="text-gray-300 hover:text-white transition-colors">About</Link>
          <Link href="/contact" className="text-gray-300 hover:text-white transition-colors">Contact</Link>
        </div>
      </nav>

      {/* Hero Section */}
      <div className="max-w-7xl mx-auto px-6 py-20">
        <div className="grid lg:grid-cols-2 gap-12 items-center">
          <div className="space-y-8">
            <div className="space-y-4">
              <div className="inline-flex items-center px-3 py-1 rounded-full bg-blue-500/10 border border-blue-500/20 text-blue-400 text-sm">
                <Zap className="h-4 w-4 mr-2" />
                AI-Powered Medical Analysis
              </div>
              <h1 className="text-5xl lg:text-7xl font-bold text-white leading-tight">
                Brain Tumor
                <span className="text-transparent bg-clip-text bg-gradient-to-r from-blue-400 to-purple-400">
                  {" "}Classification
                </span>
              </h1>
              <p className="text-xl text-gray-300 leading-relaxed">
                Advanced deep learning model for rapid and accurate brain tumor detection from MRI scans. 
                Trained on thousands of medical images with 94.2% accuracy.
              </p>
            </div>

            <div className="flex flex-col sm:flex-row gap-4">
              <Link href="/classifier">
                <Button 
                  size="lg" 
                  className="bg-blue-500 hover:bg-blue-600 text-white px-8 py-4 text-lg"
                >
                  Try Classifier
                  <ArrowRight className="ml-2 h-5 w-5" />
                </Button>
              </Link>
              <Link href="/research">
                <Button 
                  variant="outline" 
                  size="lg" 
                  className="border-gray-600 text-gray-300 hover:bg-gray-800 px-8 py-4 text-lg"
                >
                  View Research
                </Button>
              </Link>
            </div>

            {/* Trust Indicators */}
            <div className="flex items-center space-x-8 pt-8">
              <div className="text-center">
                <div className="text-2xl font-bold text-white">94.2%</div>
                <div className="text-sm text-gray-400">Accuracy</div>
              </div>
              <div className="text-center">
                <div className="text-2xl font-bold text-white">5,000+</div>
                <div className="text-sm text-gray-400">Training Images</div>
              </div>
              <div className="text-center">
                <div className="text-2xl font-bold text-white">&lt;2s</div>
                <div className="text-sm text-gray-400">Analysis Time</div>
              </div>
            </div>
          </div>

          {/* Feature Cards */}
          <div className="grid gap-6">
            <Card className="p-6 bg-gray-800/50 border-gray-700 backdrop-blur">
              <div className="flex items-start space-x-4">
                <div className="p-2 bg-blue-500/10 rounded-lg">
                  <Brain className="h-6 w-6 text-blue-400" />
                </div>
                <div>
                  <h3 className="text-lg font-semibold text-white mb-2">Deep Learning Architecture</h3>
                  <p className="text-gray-300">Convolutional Neural Network trained on diverse MRI datasets</p>
                </div>
              </div>
            </Card>

            <Card className="p-6 bg-gray-800/50 border-gray-700 backdrop-blur">
              <div className="flex items-start space-x-4">
                <div className="p-2 bg-green-500/10 rounded-lg">
                  <Shield className="h-6 w-6 text-green-400" />
                </div>
                <div>
                  <h3 className="text-lg font-semibold text-white mb-2">HIPAA Compliant</h3>
                  <p className="text-gray-300">No data stored. Images processed locally and discarded immediately</p>
                </div>
              </div>
            </Card>

            <Card className="p-6 bg-gray-800/50 border-gray-700 backdrop-blur">
              <div className="flex items-start space-x-4">
                <div className="p-2 bg-purple-500/10 rounded-lg">
                  <Zap className="h-6 w-6 text-purple-400" />
                </div>
                <div>
                  <h3 className="text-lg font-semibold text-white mb-2">Instant Results</h3>
                  <p className="text-gray-300">Real-time classification with confidence scores and explanations</p>
                </div>
              </div>
            </Card>
          </div>
        </div>
      </div>
    </div>
  );
}