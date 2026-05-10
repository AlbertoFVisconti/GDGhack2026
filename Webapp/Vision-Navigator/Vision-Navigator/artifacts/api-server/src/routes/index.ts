import { Router, type IRouter } from "express";
import healthRouter from "./health";
import cameraRouter from "./camera";

const router: IRouter = Router();

router.use(healthRouter);
router.use(cameraRouter);

export default router;
