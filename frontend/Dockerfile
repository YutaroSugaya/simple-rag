FROM node:20-alpine
WORKDIR /usr/src/app
COPY package.json package-lock.json ./
RUN npm i --legacy-peer-deps --silent
COPY . ./
CMD ["npm", "run", "dev"]
