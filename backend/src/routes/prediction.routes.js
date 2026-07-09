import { Router } from "express";
import { upload } from "../middleware/upload.js";
import { predictAudio, healthCheck } from "../controllers/prediction.controller.js";

const router = Router();

router.get("/health", healthCheck);
router.post("/predict", upload.single("audio"), predictAudio);

export default router;
