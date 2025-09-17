import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Image optimization configuration
  images: {
    // Modern image formats for better performance
    formats: ['image/webp', 'image/avif'],
    // You can also add other image optimization settings:
    // deviceSizes: [640, 750, 828, 1080, 1200, 1920, 2048, 3840],
    // imageSizes: [16, 32, 48, 64, 96, 128, 256, 384],
    // domains: ['example.com'], // if loading images from external domains
  },
};

export default nextConfig;