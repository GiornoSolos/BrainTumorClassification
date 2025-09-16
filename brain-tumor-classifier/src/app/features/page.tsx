import { Brain, Shield, Zap, Upload, BarChart3, Clock, ArrowLeft } from 'lucide-react';
import { Card } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import Link from 'next/link';

export default function FeaturesPage() {
  const features = [
    {
      icon: Brain,
      title: "Advanced Deep Learning",
      description: "ResNet50-based Convolutional Neural Network trained on over 5,000 medical images for accurate brain tumor classification.",
      specs: ["94.2% accuracy", "4-class classification", "Medical-grade performance"]
    },
    {
      icon: Zap,
      title: "Real-Time Analysis",
      description: "Instant MRI scan processing with results delivered in under 2 seconds, enabling rapid screening and assessment.",
      specs: ["< 2 second processing", "Edge computing", "Optimized inference"]
    },
    {
      icon: Shield,
      title: "Privacy-First Design",
      description: "HIPAA-compliant architecture with no data storage. Images are processed locally and immediately discarded.",
      specs: ["No data retention", "Local processing", "Secure transmission"]
    },
    {
      icon: Upload,
      title: "Intuitive Interface",
      description: "Professional drag-and-drop interface supporting multiple image formats with comprehensive file validation.",
      specs: ["Drag & drop", "Multi-format support", "10MB file limit"]
    },
    {
      icon: BarChart3,
      title: "Detailed Results",
      description: "Comprehensive analysis including confidence scores, detailed explanations, and clinical context for each prediction.",
      specs: ["Confidence scoring", "Clinical explanations", "Visual feedback"]
    },
    {
      icon: Clock,
      title: "Always Available",
      description: "24/7 accessibility through web interface with global CDN deployment for consistent performance worldwide.",
      specs: ["Global availability", "99.9% uptime", "Mobile responsive"]
    }
  ];

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-blue-50 dark:from-slate-900 dark:to-slate-800">
      {/* Navigation */}
      <nav className="flex items-center justify-between p-6 max-w-7xl mx-auto">
        <Link href="/" className="flex items-center space-x-2 text-blue-600 hover:text-blue-700">
          <ArrowLeft className="h-5 w-5" />
          <span>Back to Home</span>
        </Link>
        <div className="flex items-center space-x-2">
          <Brain className="h-8 w-8 text-blue-400" />
          <span className="text-xl font-bold text-gray-900 dark:text-white">NeuroClassify</span>
        </div>
      </nav>

      {/* Header */}
      <div className="max-w-7xl mx-auto px-6 py-12">
        <div className="text-center space-y-4 mb-16">
          <Badge variant="outline" className="text-blue-600 border-blue-200">
            Advanced Medical AI
          </Badge>
          <h1 className="text-4xl lg:text-6xl font-bold text-gray-900 dark:text-white">
            Features & Capabilities
          </h1>
          <p className="text-xl text-gray-600 dark:text-gray-300 max-w-3xl mx-auto">
            Comprehensive brain tumor classification powered by state-of-the-art deep learning technology
          </p>
        </div>

        {/* Features Grid */}
        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-8 mb-16">
          {features.map((feature, index) => (
            <Card key={index} className="p-8 bg-white dark:bg-gray-800 shadow-lg hover:shadow-xl transition-shadow">
              <div className="flex items-start space-x-4">
                <div className="p-3 bg-blue-500/10 rounded-lg flex-shrink-0">
                  <feature.icon className="h-8 w-8 text-blue-500" />
                </div>
                <div className="flex-1">
                  <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-3">
                    {feature.title}
                  </h3>
                  <p className="text-gray-600 dark:text-gray-300 mb-4 leading-relaxed">
                    {feature.description}
                  </p>
                  <div className="space-y-2">
                    {feature.specs.map((spec, specIndex) => (
                      <div key={specIndex} className="flex items-center text-sm">
                        <div className="w-1.5 h-1.5 bg-blue-500 rounded-full mr-2"></div>
                        <span className="text-gray-500 dark:text-gray-400">{spec}</span>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            </Card>
          ))}
        </div>

        {/* Technical Specifications */}
        <Card className="p-8 bg-white dark:bg-gray-800 shadow-lg">
          <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-6">
            Technical Specifications
          </h2>
          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-8">
            <div>
              <h3 className="font-semibold text-gray-900 dark:text-white mb-2">Model Architecture</h3>
              <ul className="text-gray-600 dark:text-gray-300 text-sm space-y-1">
                <li>ResNet50 base model</li>
                <li>Transfer learning</li>
                <li>Custom classification head</li>
                <li>Batch normalization</li>
              </ul>
            </div>
            <div>
              <h3 className="font-semibold text-gray-900 dark:text-white mb-2">Training Data</h3>
              <ul className="text-gray-600 dark:text-gray-300 text-sm space-y-1">
                <li>5,000+ MRI images</li>
                <li>4 tumor categories</li>
                <li>Balanced dataset</li>
                <li>Medical validation</li>
              </ul>
            </div>
            <div>
              <h3 className="font-semibold text-gray-900 dark:text-white mb-2">Performance</h3>
              <ul className="text-gray-600 dark:text-gray-300 text-sm space-y-1">
                <li>94.2% accuracy</li>
                <li>92.8% precision</li>
                <li>93.5% recall</li>
                <li>93.1% F1-score</li>
              </ul>
            </div>
            <div>
              <h3 className="font-semibold text-gray-900 dark:text-white mb-2">Classifications</h3>
              <ul className="text-gray-600 dark:text-gray-300 text-sm space-y-1">
                <li>Glioma detection</li>
                <li>Meningioma detection</li>
                <li>Pituitary tumors</li>
                <li>Normal tissue</li>
              </ul>
            </div>
          </div>
        </Card>
      </div>
    </div>
  );
}