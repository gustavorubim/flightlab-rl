FROM node:22-alpine

ENV NODE_ENV=development

WORKDIR /workspace/apps/mission-control/web

RUN corepack enable

WORKDIR /workspace

COPY apps/mission-control /workspace/apps/mission-control

WORKDIR /workspace/apps/mission-control/web

RUN npm install

CMD ["npm", "run", "dev", "--", "--host", "0.0.0.0", "--port", "5173"]
