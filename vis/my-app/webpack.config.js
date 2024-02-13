const resolve = require('path').resolve;
const join = require('path').join;
const webpack = require('webpack');

const CONFIG = {
  // bundle app.js and everything it imports, recursively.
  entry: {
    app: resolve('./src/main.js')
  },

  devtool: 'source-map',

  module: {
    rules: [
      {
        test: /\.js$/,
        loader: 'babel-loader',
        include: join(__dirname, 'src'),
        exclude: [/node_modules/]
      },
      {
        // The example has some JSON data
        test: /\.json$/,
        loader: 'json-loader',
        exclude: [/node_modules/]
      }
    ]
  },

  node: {
    fs: 'empty'
  },

  // Optional: Enables reading mapbox token from environment variable
  plugins: [
    new webpack.EnvironmentPlugin(['MapboxAccessToken'])
  ]
};

// This line enables bundling against src in this repo rather than installed deck.gl module
module.exports = CONFIG;