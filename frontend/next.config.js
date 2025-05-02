const nextConfig = {
  reactStrictMode: true,
  output: 'standalone',
  // バックエンドサーバーへのプロキシ設定を追加
  async rewrites() {
    return [
      {
        source: '/api/:path*',
        destination: 'http://backend:8000/:path*',
      },
    ];
  },
};

module.exports = nextConfig;
