import { Brain, Mail, Github, Linkedin, ExternalLink, ArrowLeft } from 'lucide-react';
import { Card } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import Link from 'next/link';

export default function ContactPage() {
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
            Get In Touch
          </Badge>
          <h1 className="text-4xl lg:text-6xl font-bold text-gray-900 dark:text-white">
            Contact & Connect
          </h1>
          <p className="text-xl text-gray-600 dark:text-gray-300 max-w-3xl mx-auto">
            Questions about the brain tumor classification project, collaboration opportunities, 
            or feedback on the ResNet50 implementation?
          </p>
        </div>

        <div className="grid md:grid-cols-3 gap-8 mb-16">
          <Card className="p-8 bg-white dark:bg-gray-800 shadow-lg hover:shadow-xl transition-shadow">
            <div className="text-center">
              <div className="p-4 bg-blue-500/10 rounded-lg w-fit mx-auto mb-4">
                <Mail className="h-8 w-8 text-blue-500" />
              </div>
              <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-3">
                Email
              </h3>
              <p className="text-gray-600 dark:text-gray-300 mb-6">
                For project questions, collaboration, or technical discussions
              </p>
              <Button variant="outline" asChild className="w-full">
                <a href="mailto:your.email@example.com">
                  your.email@example.com
                  <ExternalLink className="ml-2 h-4 w-4" />
                </a>
              </Button>
            </div>
          </Card>

          <Card className="p-8 bg-white dark:bg-gray-800 shadow-lg hover:shadow-xl transition-shadow">
            <div className="text-center">
              <div className="p-4 bg-green-500/10 rounded-lg w-fit mx-auto mb-4">
                <Github className="h-8 w-8 text-green-500" />
              </div>
              <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-3">
                GitHub
              </h3>
              <p className="text-gray-600 dark:text-gray-300 mb-6">
                View source code, contribute, or report issues
              </p>
              <Button variant="outline" asChild className="w-full">
                <a href="https://github.com/yourusername/BrainTumorClassification" target="_blank" rel="noopener noreferrer">
                  View Repository
                  <ExternalLink className="ml-2 h-4 w-4" />
                </a>
              </Button>
            </div>
          </Card>

          <Card className="p-8 bg-white dark:bg-gray-800 shadow-lg hover:shadow-xl transition-shadow">
            <div className="text-center">
              <div className="p-4 bg-purple-500/10 rounded-lg w-fit mx-auto mb-4">
                <Linkedin className="h-8 w-8 text-purple-500" />
              </div>
              <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-3">
                LinkedIn
              </h3>
              <p className="text-gray-600 dark:text-gray-300 mb-6">
                Professional networking and opportunities
              </p>
              <Button variant="outline" asChild className="w-full">
                <a href="https://linkedin.com/in/yourprofile" target="_blank" rel="noopener noreferrer">
                  Connect
                  <ExternalLink className="ml-2 h-4 w-4" />
                </a>
              </Button>
            </div>
          </Card>
        </div>

        <Card className="p-8 mb-16 bg-white dark:bg-gray-800 shadow-lg">
          <h2 className="text-3xl font-bold text-gray-900 dark:text-white mb-8 text-center">
            Frequently Asked Questions
          </h2>
          <div className="space-y-6">
            <div>
              <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-2">
                How accurate is the brain tumor classification model?
              </h3>
              <p className="text-gray-600 dark:text-gray-300">
                The ResNet50-based model achieves 94.2% accuracy on the test dataset with comprehensive 
                evaluation across all four tumor types (glioma, meningioma, normal tissue, pituitary).
              </p>
            </div>
            <div>
              <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-2">
                Is medical data stored by this application?
              </h3>
              <p className="text-gray-600 dark:text-gray-300">
                No medical images or personal data are stored. All uploaded images are processed 
                locally and immediately discarded after analysis, ensuring complete privacy protection.
              </p>
            </div>
            <div>
              <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-2">
                Can this tool be used for actual medical diagnosis?
              </h3>
              <p className="text-gray-600 dark:text-gray-300">
                No, this is a research and educational demonstration only. It should not be used 
                for medical diagnosis or treatment decisions. Always consult qualified healthcare professionals.
              </p>
            </div>
            <div>
              <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-2">
                What technologies power this application?
              </h3>
              <p className="text-gray-600 dark:text-gray-300">
                The backend uses PyTorch for model training, converted to ONNX for web deployment. 
                The frontend is built with Next.js, TypeScript, and Tailwind CSS, deployed on Vercel's edge network.
              </p>
            </div>
          </div>
        </Card>

        <Card className="p-6 bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800">
          <div className="text-center">
            <h2 className="text-2xl font-bold text-blue-800 dark:text-blue-200 mb-4">
              Technical Support
            </h2>
            <p className="text-blue-700 dark:text-blue-300 mb-6">
              Experiencing technical issues with the brain tumor classifier or have suggestions for model improvements?
            </p>
            <Button asChild className="bg-blue-600 hover:bg-blue-700 text-white">
              <a href="https://github.com/yourusername/BrainTumorClassification/issues" target="_blank" rel="noopener noreferrer">
                <Github className="mr-2 h-4 w-4" />
                Report Issues
                <ExternalLink className="ml-2 h-4 w-4" />
              </a>
            </Button>
          </div>
        </Card>
      </div>
    </div>
  );
}