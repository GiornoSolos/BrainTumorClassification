import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  experimental: {
    serverComponentsExternalPackages: ['onnxruntime-node', 'sharp'],
  },
  
  // Optimize for Vercel serverless functions
  webpack: (config, { dev, isServer }) => {
    if (isServer) {
      // Externalize onnxruntime-node to prevent bundling issues
      config.externals = config.externals || [];
      config.externals.push('onnxruntime-node');
      
      // Configure memory management
      config.optimization = config.optimization || {};
      config.optimization.nodeEnv = false;
    }
    
    return config;
  },
  
  // API route configuration for better memory management
  api: {
    responseLimit: false,
  },
  
  // Reduce build size and memory usage
  swcMinify: true,
  
  // Image optimization
  images: {
    formats: ['image/webp', 'image/avif'],
  },
};

export default nextConfig;