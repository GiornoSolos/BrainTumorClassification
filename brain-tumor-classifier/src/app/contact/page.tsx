import { Brain, Mail, Github, Linkedin, ExternalLink, ArrowLeft } from 'lucide-react';
import { Card } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import Link from 'next/link';

export default function ContactPage() {
  const contactMethods = [
    {
      icon: Mail,
      title: "Email",
      description: "Get in touch for collaborations, questions, or feedback",
      contact: "your.email@example.com",
      href: "mailto:your.email@example.com"
    },
    {
      icon: Linkedin,
      title: "LinkedIn",
      description: "Connect for professional networking and opportunities",
      contact: "Your LinkedIn Profile",
      href: "https://linkedin.com/in/yourprofile"
    },
    {
      icon: Github,
      title: "GitHub",
      description: "View source code, contribute, or report issues",
      contact: "github.com/yourusername",
      href: "https://github.com/yourusername"
    }
  ];

  const projectInfo = [
    {
      title: "Open Source",
      description: "NeuroClassify is built with open-source technologies and follows best practices for medical AI development."
    },
    {
      title: "Research Focus",
      description: "This project demonstrates practical applications of deep learning in medical imaging and computer vision."
    },
    {
      title: "Educational Purpose",
      description: "Designed to showcase modern web development, machine learning integration, and healthcare technology."
    },
    {
      title: "Collaboration Welcome",
      description: "Open to feedback, contributions, and discussions about improving medical AI applications."
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
            Get In Touch
          </Badge>
          <h1 className="text-4xl lg:text-6xl font-bold text-gray-900 dark:text-white">
            Contact & Connect
          </h1>
          <p className="text-xl text-gray-600 dark:text-gray-300 max-w-3xl mx-auto">
            Questions about the project, collaboration opportunities, or feedback on the platform? 
            I would love to hear from you.
          </p>
        </div>

        {/* Contact Methods */}
        <div className="grid md:grid-cols-3 gap-8 mb-16">
          {contactMethods.map((method, index) => (
            <Card key={index} className="p-8 bg-white dark:bg-gray-800 shadow-lg hover:shadow-xl transition-shadow">
              <div className="text-center">
                <div className="p-4 bg-blue-500/10 rounded-lg w-fit mx-auto mb-4">
                  <method.icon className="h-8 w-8 text-blue-500" />
                </div>
                <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-3">
                  {method.title}
                </h3>
                <p className="text-gray-600 dark:text-gray-300 mb-6">
                  {method.description}
                </p>
                <Button 
                  variant="outline" 
                  asChild
                  className="w-full"
                >
                  <a href={method.href} target="_blank" rel="noopener noreferrer">
                    {method.contact}
                    <ExternalLink className="ml-2 h-4 w-4" />
                  </a>
                </Button>
              </div>
            </Card>
          ))}
        </div>

        {/* Project Information */}
        <Card className="p-8 mb-16 bg-white dark:bg-gray-800 shadow-lg">
          <h2 className="text-3xl font-bold text-gray-900 dark:text-white mb-8 text-center">
            About This Project
          </h2>
          <div className="grid md:grid-cols-2 gap-8">
            {projectInfo.map((info, index) => (
              <div key={index} className="flex items-start space-x-4">
                <div className="w-2 h-2 bg-blue-500 rounded-full mt-3 flex-shrink-0"></div>
                <div>
                  <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-2">
                    {info.title}
                  </h3>
                  <p className="text-gray-600 dark:text-gray-300">
                    {info.description}
                  </p>
                </div>
              </div>
            ))}
          </div>
        </Card>

        {/* FAQ Section */}
        <Card className="p-8 mb-16 bg-white dark:bg-gray-800 shadow-lg">
          <h2 className="text-3xl font-bold text-gray-900 dark:text-white mb-8 text-center">
            Frequently Asked Questions
          </h2>
          <div className="space-y-6">
            <div>
              <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-2">
                Can this tool be used for actual medical diagnosis?
              </h3>
              <p className="text-gray-600 dark:text-gray-300">
                No, NeuroClassify is designed for research and educational purposes only. It should not be used 
                for medical diagnosis or treatment decisions. Always consult qualified healthcare professionals.
              </p>
            </div>
            <div>
              <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-2">
                How accurate is the brain tumor classification?
              </h3>
              <p className="text-gray-600 dark:text-gray-300">
                The model achieves 94.2% accuracy on the test dataset. However, performance may vary with 
                different image qualities, formats, or medical conditions not represented in the training data.
              </p>
            </div>
            <div>
              <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-2">
                Is my medical data stored or shared?
              </h3>
              <p className="text-gray-600 dark:text-gray-300">
                No data is stored. All uploaded images are processed locally and immediately discarded after 
                analysis. The platform follows privacy-first principles with no data retention.
              </p>
            </div>
            <div>
              <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-2">
                Can I contribute to the project?
              </h3>
              <p className="text-gray-600 dark:text-gray-300">
                Yes! The project welcomes contributions, feedback, and suggestions. Visit the GitHub repository 
                to view the source code, report issues, or submit improvements.
              </p>
            </div>
          </div>
        </Card>

        {/* Technical Support */}
        <Card className="p-8 bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800">
          <div className="text-center">
            <h2 className="text-2xl font-bold text-blue-800 dark:text-blue-200 mb-4">
              Technical Support
            </h2>
            <p className="text-blue-700 dark:text-blue-300 mb-6">
              Experiencing technical issues or have suggestions for improvements? 
              Report bugs or request features through GitHub Issues.
            </p>
            <Button asChild className="bg-blue-600 hover:bg-blue-700 text-white">
              <a href="https://github.com/yourusername/BrainTumorClassification/issues" target="_blank" rel="noopener noreferrer">
                <Github className="mr-2 h-4 w-4" />
                Report Issues on GitHub
                <ExternalLink className="ml-2 h-4 w-4" />
              </a>
            </Button>
          </div>
        </Card>
      </div>
    </div>
  );
}