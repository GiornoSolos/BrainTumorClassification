import type { Metadata } from 'next';
import { Inter } from 'next/font/google';
import './globals.css';

const inter = Inter({ subsets: ['latin'] });

export const metadata: Metadata = {
  title: {
    default: 'NeuroClassify - AI Brain Tumor Detection',
    template: '%s | NeuroClassify'
  },
  description: 'Advanced AI-powered brain tumor classification from MRI scans. Fast, accurate, and secure medical image analysis with 94.2% accuracy.',
  keywords: [
    'brain tumor detection',
    'MRI analysis', 
    'medical AI',
    'deep learning',
    'healthcare technology',
    'neural networks',
    'medical imaging',
    'CNN',
    'TensorFlow',
    'machine learning'
  ],
  authors: [{ name: 'Your Name' }],
  creator: 'Your Name',
  publisher: 'NeuroClassify',
  robots: {
    index: true,
    follow: true,
    googleBot: {
      index: true,
      follow: true,
      'max-video-preview': -1,
      'max-image-preview': 'large',
      'max-snippet': -1,
    },
  },
  openGraph: {
    type: 'website',
    locale: 'en_US',
    url: 'https://your-domain.vercel.app',
    title: 'NeuroClassify - AI Brain Tumor Detection',
    description: 'Advanced AI-powered brain tumor classification from MRI scans with 94.2% accuracy.',
    siteName: 'NeuroClassify',
  },
  twitter: {
    card: 'summary_large_image',
    title: 'NeuroClassify - AI Brain Tumor Detection',
    description: 'Advanced AI-powered brain tumor classification from MRI scans.',
    creator: '@yourtwitterhandle',
  },
  category: 'technology',
  viewport: {
    width: 'device-width',
    initialScale: 1,
    maximumScale: 1,
  },
  icons: {
    icon: '/favicon.ico',
    shortcut: '/favicon-16x16.png',
    apple: '/apple-touch-icon.png',
  },
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="scroll-smooth">
      <head>
        <script
          type="application/ld+json"
          dangerouslySetInnerHTML={{
            __html: JSON.stringify({
              '@context': 'https://schema.org',
              '@type': 'WebApplication',
              name: 'NeuroClassify',
              description: 'AI-powered brain tumor classification from MRI scans',
              applicationCategory: 'MedicalApplication',
              operatingSystem: 'Web',
              offers: {
                '@type': 'Offer',
                price: '0',
                priceCurrency: 'USD'
              },
              author: {
                '@type': 'Person',
                name: 'Your Name'
              }
            })
          }}
        />
      </head>
      <body className={`${inter.className} antialiased`}>
        <div className="min-h-screen">
          {children}
        </div>
      </body>
    </html>
  );
}