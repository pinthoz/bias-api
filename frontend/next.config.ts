import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Static site: `next build` writes plain HTML/JS/CSS to ./out, ready for S3
  output: "export",
  // Every page becomes folder/index.html, which S3 + CloudFront serve without extra rules
  trailingSlash: true,
  // The image optimiser needs a server; a static export has none
  images: { unoptimized: true },
};

export default nextConfig;
