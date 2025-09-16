import { Brain, Target, Users, Award, ArrowLeft } from 'lucide-react';
import { Card } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import Link from 'next/link';

export default function AboutPage() {
  const stats = [
    { label: "Model Accuracy", value: "94.2%" },
    { label: "Training Images", value: "5,000+" },
    { label: "Processing Time", value: "< 2s" },
    { label: "Classifications", value: "4 Types" }
  ];

  const timeline = [
    {
      year: "2024",
      title: "Research & Development",
      description: "Extensive research into medical imaging AI and brain tumor classification methodologies."
    },
    {
      year: "2024",
      title: "Model Training",
      description: "Development and training of ResNet50-based CNN on comprehensive brain tumor dataset."
    },
    {
      year: "2024",
      title: "Web Application",
      description: "Creation of professional web interface with modern React and Next.js architecture."
    },
    {
      year: "2024",
      title: "Production Deployment",
      description: "Launch of NeuroClassify platform with global CDN and secure processing capabilities."
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

      {/* Hero Section */}
      <div className="max-w-7xl mx-auto px-6 py-12">
        <div className="text-center space-y-4 mb-16">
          <Badge variant="outline" className="text-blue-600 border-blue-200">
            Medical AI Innovation
          </Badge>
          <h1 className="text-4xl lg:text-6xl font-bold text-gray-900 dark:text-white">
            About NeuroClassify
          </h1>
          <p className="text-xl text-gray-600 dark:text-gray-300 max-w-3xl mx-auto">
            Advancing medical imaging through artificial intelligence to support healthcare professionals 
            in brain tumor detection and classification
          </p>
        </div>

        {/* Mission Statement */}
        <Card className="p-8 mb-16 bg-white dark:bg-gray-800 shadow-lg">
          <div className="grid lg:grid-cols-2 gap-8 items-center">
            <div>
              <h2 className="text-3xl font-bold text-gray-900 dark:text-white mb-4">
                Our Mission
              </h2>
              <p className="text-gray-600 dark:text-gray-300 leading-relaxed mb-6">
                NeuroClassify was developed to demonstrate the potential of artificial intelligence 
                in medical imaging analysis. Our goal is to create accessible, accurate, and 
                privacy-conscious tools that can assist in educational and research contexts.
              </p>
              <p className="text-gray-600 dark:text-gray-300 leading-relaxed">
                While this platform serves as a research demonstration, we envision a future where 
                AI-powered medical tools enhance healthcare delivery worldwide, providing rapid 
                screening capabilities and supporting clinical decision-making.
              </p>
            </div>
            <div className="grid grid-cols-2 gap-4">
              {stats.map((stat, index) => (
                <div key={index} className="text-center p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
                  <div className="text-2xl font-bold text-blue-600 dark:text-blue-400">
                    {stat.value}
                  </div>
                  <div className="text-sm text-gray-500 dark:text-gray-400">
                    {stat.label}
                  </div>
                </div>
              ))}
            </div>
          </div>
        </Card>

        {/* Key Principles */}
        <div className="grid md:grid-cols-3 gap-8 mb-16">
          <Card className="p-6 bg-white dark:bg-gray-800 shadow-lg text-center">
            <div className="p-3 bg-blue-500/10 rounded-lg w-fit mx-auto mb-4">
              <Target className="h-8 w-8 text-blue-500" />
            </div>
            <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-3">
              Accuracy First
            </h3>
            <p className="text-gray-600 dark:text-gray-300">
              Rigorous model validation and testing to ensure reliable performance 
              across diverse medical imaging scenarios.
            </p>
          </Card>

          <Card className="p-6 bg-white dark:bg-gray-800 shadow-lg text-center">
            <div className="p-3 bg-green-500/10 rounded-lg w-fit mx-auto mb-4">
              <Users className="h-8 w-8 text-green-500" />
            </div>
            <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-3">
              Privacy Protection
            </h3>
            <p className="text-gray-600 dark:text-gray-300">
              No data storage or retention. All medical images are processed locally 
              and immediately discarded after analysis.
            </p>
          </Card>

          <Card className="p-6 bg-white dark:bg-gray-800 shadow-lg text-center">
            <div className="p-3 bg-purple-500/10 rounded-lg w-fit mx-auto mb-4">
              <Award className="h-8 w-8 text-purple-500" />
            </div>
            <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-3">
              Research Excellence
            </h3>
            <p className="text-gray-600 dark:text-gray-300">
              Built on established medical research and validated against 
              peer-reviewed standards in medical imaging AI.
            </p>
          </Card>
        </div>

        {/* Development Timeline */}
        <Card className="p-8 mb-16 bg-white dark:bg-gray-800 shadow-lg">
          <h2 className="text-3xl font-bold text-gray-900 dark:text-white mb-8 text-center">
            Development Timeline
          </h2>
          <div className="space-y-8">
            {timeline.map((item, index) => (
              <div key={index} className="flex items-start space-x-4">
                <div className="flex-shrink-0">
                  <div className="w-12 h-12 bg-blue-500 rounded-full flex items-center justify-center">
                    <span className="text-white font-semibold text-sm">{item.year}</span>
                  </div>
                </div>
                <div className="flex-1 min-w-0">
                  <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
                    {item.title}
                  </h3>
                  <p className="text-gray-600 dark:text-gray-300 mt-1">
                    {item.description}
                  </p>
                </div>
              </div>
            ))}
          </div>
        </Card>

        {/* Technology Stack */}
        <Card className="p-8 bg-white dark:bg-gray-800 shadow-lg">
          <h2 className="text-3xl font-bold text-gray-900 dark:text-white mb-8 text-center">
            Technology Stack
          </h2>
          <div className="grid md:grid-cols-3 gap-8">
            <div>
              <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
                Machine Learning
              </h3>
              <ul className="space-y-2 text-gray-600 dark:text-gray-300">
                <li>PyTorch Deep Learning Framework</li>
                <li>ResNet50 Architecture</li>
                <li>Transfer Learning</li>
                <li>Medical Image Preprocessing</li>
                <li>Model Optimization</li>
              </ul>
            </div>
            <div>
              <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
                Frontend
              </h3>
              <ul className="space-y-2 text-gray-600 dark:text-gray-300">
                <li>Next.js 15 React Framework</li>
                <li>TypeScript</li>
                <li>Tailwind CSS</li>
                <li>Shadcn/ui Components</li>
                <li>Framer Motion</li>
              </ul>
            </div>
            <div>
              <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
                Infrastructure
              </h3>
              <ul className="space-y-2 text-gray-600 dark:text-gray-300">
                <li>Vercel Edge Network</li>
                <li>Global CDN Distribution</li>
                <li>SSL Certificate</li>
                <li>Edge Runtime</li>
                <li>Automatic Deployments</li>
              </ul>
            </div>
          </div>
        </Card>

        {/* Important Disclaimer */}
        <Card className="p-6 mt-8 bg-yellow-50 dark:bg-yellow-900/20 border border-yellow-200 dark:border-yellow-800">
          <h3 className="text-lg font-semibold text-yellow-800 dark:text-yellow-200 mb-2">
            Important Medical Disclaimer
          </h3>
          <p className="text-yellow-700 dark:text-yellow-300 text-sm">
            NeuroClassify is designed for research and educational purposes only. This tool is not 
            intended for medical diagnosis, treatment recommendations, or clinical decision-making. 
            Always consult qualified healthcare professionals for medical advice, diagnosis, and treatment. 
            The predictions provided by this system should not replace professional medical judgment.
          </p>
        </Card>
      </div>
    </div>
  );
}