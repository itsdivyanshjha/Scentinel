FROM node:18-alpine

WORKDIR /app

# Copy package.json and package-lock.json
COPY package*.json ./

# Install dependencies
RUN npm install

# Copy the rest of the code
COPY . .

# Clean up any macOS resource fork files that cause UTF-8 encoding errors on Windows
# This removes files starting with ._ which are macOS metadata files
RUN find . -name "._*" -type f -delete || true
RUN find . -name ".DS_Store" -type f -delete || true

# Remove any existing node_modules and reinstall to ensure correct line endings
# This prevents issues with Node.js binaries corrupted by Windows line endings
RUN rm -rf node_modules && npm install

# Build the app
RUN npm run build

# Expose the port
EXPOSE 3000

# Start the app
CMD ["npm", "run", "dev"] 