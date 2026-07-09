import express from "express";
import cors from "cors";
import { PORT } from "./src/config/config.js";
import predictionRoutes from "./src/routes/prediction.routes.js";
import insightsRoutes from "./src/routes/insights.routes.js";

const app = express();

app.use(cors());
app.use(express.json());

app.use("/api", predictionRoutes);
app.use("/api", insightsRoutes);

// Central error handler — multer and controller errors land here
app.use((err, req, res, next) => {
  console.error(err);
  res.status(err.status || 500).json({ error: err.message || "Internal server error" });
});

app.listen(PORT, () => {
  console.log(`AffectCare backend listening on http://localhost:${PORT}`);
});
