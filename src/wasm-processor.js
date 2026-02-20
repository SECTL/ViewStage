/**
 * WASM 点处理模块
 * 提供高性能的点优化、距离计算等功能
 */

class WasmPointProcessor {
    constructor() {
        this.wasmModule = null;
        this.isLoaded = false;
        this.loadPromise = null;
    }

    async load() {
        if (this.isLoaded) {
            return this.wasmModule;
        }

        if (this.loadPromise) {
            return this.loadPromise;
        }

        this.loadPromise = this._loadWasm();
        try {
            return await this.loadPromise;
        } catch (error) {
            this.loadPromise = null;
            throw error;
        }
    }

    async _loadWasm() {
        try {
            const { default: init, distance, quantize_coord, perpendicular_distance, simplify_points, process_stroke_points, batch_process_strokes, optimize_draw_commands, apply_image_filter, batch_apply_image_filter, transform_points, detect_collision, calculate_distance_field, collect_points, batch_process_draw_commands, adjust_color, convert_color, smooth_path, complex_collision_detection, cull_strokes_by_viewport, calculate_stroke_bounds, detect_eraser_collision, batch_process_strokes_optimized } = await import('./wasm/wasm_viewstage.js');
            
            await init();
            
            this.wasmModule = {
                distance,
                quantize_coord,
                perpendicular_distance,
                simplify_points,
                process_stroke_points,
                batch_process_strokes,
                optimize_draw_commands,
                apply_image_filter,
                batch_apply_image_filter,
                transform_points,
                detect_collision,
                calculate_distance_field,
                collect_points,
                batch_process_draw_commands,
            adjust_color,
            convert_color,
            smooth_path,
            complex_collision_detection,
            cull_strokes_by_viewport,
            calculate_stroke_bounds,
            detect_eraser_collision,
            batch_process_strokes_optimized
        };
        
        this.isLoaded = true;
        console.log('WASM 点处理模块已加载');
        
        return this.wasmModule;
    } catch (error) {
        console.error('WASM 加载失败:', error);
        throw error;
    }
    }

    async distance(x1, y1, x2, y2) {
        await this.load();
        return this.wasmModule.distance(x1, y1, x2, y2);
    }

    async quantizeCoord(coord, step) {
        await this.load();
        return this.wasmModule.quantize_coord(coord, step);
    }

    async perpendicularDistance(px, py, x1, y1, x2, y2) {
        await this.load();
        return this.wasmModule.perpendicular_distance(px, py, x1, y1, x2, y2);
    }

    async simplifyPoints(points, epsilon) {
        await this.load();
        const pointsJson = JSON.stringify(points);
        const resultJson = this.wasmModule.simplify_points(pointsJson, epsilon);
        return JSON.parse(resultJson);
    }

    async processStrokePoints(points, config) {
        await this.load();
        const request = {
            points,
            config
        };
        const requestJson = JSON.stringify(request);
        const resultJson = this.wasmModule.process_stroke_points(requestJson);
        return JSON.parse(resultJson);
    }

    async batchProcessStrokes(strokes, config) {
        await this.load();
        const request = {
            strokes,
            config
        };
        const requestJson = JSON.stringify(request);
        const resultJson = this.wasmModule.batch_process_strokes(requestJson);
        return JSON.parse(resultJson);
    }

    async optimizeDrawCommands(commands, canvasWidth, canvasHeight) {
        await this.load();
        const request = {
            commands,
            canvas_width: canvasWidth,
            canvas_height: canvasHeight
        };
        const requestJson = JSON.stringify(request);
        const resultJson = this.wasmModule.optimize_draw_commands(requestJson);
        return JSON.parse(resultJson);
    }

    async applyImageFilter(imageData, brightness = 10, contrast = 1.4, saturation = 1.2) {
        await this.load();
        const request = {
            image_data: imageData,
            brightness,
            contrast,
            saturation
        };
        const requestJson = JSON.stringify(request);
        return this.wasmModule.apply_image_filter(requestJson);
    }

    async batchApplyImageFilter(images, brightness = 10, contrast = 1.4, saturation = 1.2) {
        await this.load();
        const request = {
            images,
            brightness,
            contrast,
            saturation
        };
        const requestJson = JSON.stringify(request);
        const resultJson = this.wasmModule.batch_apply_image_filter(requestJson);
        return JSON.parse(resultJson);
    }

    async transformPoints(points, matrix) {
        await this.load();
        const request = {
            points,
            matrix
        };
        const requestJson = JSON.stringify(request);
        const resultJson = this.wasmModule.transform_points(requestJson);
        return JSON.parse(resultJson);
    }

    async detectCollision(stroke, rect) {
        await this.load();
        const request = {
            stroke,
            rect
        };
        const requestJson = JSON.stringify(request);
        return this.wasmModule.detect_collision(requestJson);
    }

    async calculateDistanceField(points, width, height) {
        await this.load();
        const request = {
            points,
            width,
            height
        };
        const requestJson = JSON.stringify(request);
        const resultJson = this.wasmModule.calculate_distance_field(requestJson);
        return JSON.parse(resultJson);
    }

    async collectPoints(points, config, lastTime, lastX, lastY) {
        await this.load();
        const currentTime = Date.now();
        const request = {
            points,
            config,
            lastTime: lastTime,
            lastX: lastX,
            lastY: lastY,
            currentTime: currentTime
        };
        const requestJson = JSON.stringify(request);
        const resultJson = this.wasmModule.collect_points(requestJson);
        return JSON.parse(resultJson);
    }

    async batchProcessDrawCommands(commands, minDistance, maxBatchSize) {
        await this.load();
        const request = {
            commands,
            min_distance: minDistance,
            max_batch_size: maxBatchSize
        };
        const requestJson = JSON.stringify(request);
        const resultJson = this.wasmModule.batch_process_draw_commands(requestJson);
        return JSON.parse(resultJson);
    }

    // 颜色处理方法
    async adjustColor(color, brightness, contrast, saturation) {
        await this.load();
        const request = {
            color,
            brightness,
            contrast,
            saturation
        };
        const requestJson = JSON.stringify(request);
        const resultJson = this.wasmModule.adjust_color(requestJson);
        return JSON.parse(resultJson);
    }

    async convertColor(color, targetFormat) {
        await this.load();
        const request = {
            color,
            target_format: targetFormat
        };
        const requestJson = JSON.stringify(request);
        const resultJson = this.wasmModule.convert_color(requestJson);
        return JSON.parse(resultJson);
    }

    // 路径平滑方法
    async smoothPath(points, smoothness, algorithm = 'bezier') {
        await this.load();
        const request = {
            points,
            smoothness,
            algorithm
        };
        const requestJson = JSON.stringify(request);
        const resultJson = this.wasmModule.smooth_path(requestJson);
        return JSON.parse(resultJson);
    }

    async complexCollisionDetection(shape1Type, shape1Data, shape2Type, shape2Data) {
        await this.load();
        const request = {
            shape1_type: shape1Type,
            shape1_data: shape1Data,
            shape2_type: shape2Type,
            shape2_data: shape2Data
        };
        const requestJson = JSON.stringify(request);
        return this.wasmModule.complex_collision_detection(requestJson);
    }

    async cullStrokesByViewport(strokes, viewport) {
        await this.load();
        const request = {
            strokes,
            viewport
        };
        const requestJson = JSON.stringify(request);
        const resultJson = this.wasmModule.cull_strokes_by_viewport(requestJson);
        return JSON.parse(resultJson);
    }

    async calculateStrokeBounds(stroke) {
        await this.load();
        const strokeJson = JSON.stringify(stroke);
        const resultJson = this.wasmModule.calculate_stroke_bounds(strokeJson);
        return JSON.parse(resultJson);
    }

    async detectEraserCollision(strokes, eraserStroke, tolerance) {
        await this.load();
        const request = {
            strokes,
            eraser_stroke: eraserStroke,
            tolerance
        };
        const requestJson = JSON.stringify(request);
        const resultJson = this.wasmModule.detect_eraser_collision(requestJson);
        return JSON.parse(resultJson);
    }

    async batchProcessStrokesOptimized(strokes, config, viewport = null) {
        await this.load();
        const request = {
            strokes,
            config,
            viewport
        };
        const requestJson = JSON.stringify(request);
        const resultJson = this.wasmModule.batch_process_strokes_optimized(requestJson);
        return JSON.parse(resultJson);
    }
}

// 导出单例实例
const wasmPointProcessor = new WasmPointProcessor();
export default wasmPointProcessor;