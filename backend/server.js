import express from "express";
import cors from "cors";
import { PORT } from "./src/config/config.js";
import predictionRoutes from "./src/routes/prediction.routes.js";
import insightsRoutes from "./src/routes/insights.routes.js";

const app = express();

// Configure CORS to allow requests from your Cloudflare Pages frontend
app.use(
  cors({
    origin: ["https://affectcare-extended.pages.dev", "http://localhost:5173"], // Add your Cloudflare Pages domain and local dev URL
    methods: ["GET", "POST"],
    allowedHeaders: ["Content-Type"],
  }),
);

app.use(express.json());

app.use("/api", predictionRoutes);
app.use("/api", insightsRoutes);

// Central error handler — multer and controller errors land here
app.use((err, req, res, next) => {
  console.error(err);
  res
    .status(err.status || 500)
    .json({ error: err.message || "Internal server error" });
});

const port = process.env.PORT || 7860;
app.listen(port, "0.0.0.0", () => {
  console.log(`AffectCare backend listening on port ${port}`);
});
