import { Router } from "express";
import { upload } from "../middleware/upload.js";
import {
  getMetrics,
  getLossHistory,
  visualizeAudio,
} from "../controllers/insights.controller.js";

const router = Router();

router.get("/model/metrics", getMetrics);
router.get("/model/loss-history", getLossHistory);
router.post("/visualize", upload.single("audio"), visualizeAudio);

export default router;
