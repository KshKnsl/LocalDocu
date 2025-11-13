const express = require('express');
const cors = require('cors');

const app = express();
app.use(cors());
app.use(express.json());

const progressData = {};

app.get('/', (req, res) => res.json({ 
  service: 'Progress Tracking', 
  active: Object.keys(progressData).length 
}));

app.post('/progress', (req, res) => {
  const { documentId, ...data } = req.body;
  if (!documentId) return res.status(400).json({ error: 'documentId required' });
  progressData[documentId] = { documentId, ...data, timestamp: new Date().toISOString() };
  res.json({ status: 'success' });
});

app.get('/progress', (req, res) => {
  const { documentId } = req.query;
  res.json({ progress: documentId ? progressData[documentId] || null : progressData });
});

app.delete('/progress/:documentId', (req, res) => {
  const exists = delete progressData[req.params.documentId];
  res.status(exists ? 200 : 404).json({ status: exists ? 'cleared' : 'not_found' });
});

app.get('/health', (req, res) => res.json({ status: 'healthy' }));

if (require.main === module) {
  app.listen(process.env.PORT || 5000, () => console.log('Progress service running'));
}

module.exports = app;
