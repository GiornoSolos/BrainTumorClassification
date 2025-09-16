import { Brain, Shield, Zap, Upload, BarChart3, Clock, ArrowLeft } from 'lucide-react';
import { Card } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import Link from 'next/link';

export default function FeaturesPage() {
  const features = [
    {
      icon: Brain,
      title: "Advanced Deep Learning",
      description: "ResNet50-based CNN with enhanced classifier achieving 94.2% accuracy on brain tumor classification.",
      specs: ["ResNet50 backbone", "Enhanced classifier", "94.2% accuracy"]
    },
    {
      icon: Zap,
      title: "Real-Time Analysis",
      description: "Instant MRI scan processing with results delivered in under 2 seconds using optimized inference.",
      specs: ["< 2 second processing", "ONNX optimization", "Edge deployment"]
    },
    {
      icon: Shield,
      title: "Privacy-First Design",
      description: "HIPAA-compliant architecture with no data storage. Images processed locally and discarded immediately.",
      specs: ["No data retention", "Local processing", "Secure inference"]
    },
    {
      icon: Upload,
      title: "Professional Interface",
      description: "Drag-and-drop interface supporting multiple formats with comprehensive validation and error handling.",
      specs: ["Drag & drop", "Multi-format", "10MB limit"]
    },
    {
      icon: BarChart3,
      title: "Detailed Results",
      description: "Comprehensive analysis with confidence scores, probability distributions, and medical explanations.",
      specs: ["Confidence scoring", "All probabilities", "Medical context"]
    },
    {
      icon: Clock,
      title: "Production Ready",
      description: "Deployed on Vercel edge network with global CDN for consistent worldwide performance.",
      specs: ["Global deployment", "Edge computing", "Auto-scaling"]
    }
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
          <span className="text-xl font-bold text-gray-900 dark:text-white">NeuroClassify</span>
        </div>
      </nav>

      <div className="max-w-7xl mx-auto px-6 py-12">
        <div className="text-center space-y-4 mb-16">
          <Badge variant="outline" className="text-blue-600 border-blue-200">
            Advanced Medical AI
          </Badge>
          <h1 className="text-4xl lg:text-6xl font-bold text-gray-900 dark:text-white">
            Features & Capabilities
          </h1>
          <p className="text-xl text-gray-600 dark:text-gray-300 max-w-3xl mx-auto">
            Professional brain tumor classification powered by trained ResNet50 model with real-world medical accuracy
          </p>
        </div>

        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-8">
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
      </div>
    </div>
  );
}