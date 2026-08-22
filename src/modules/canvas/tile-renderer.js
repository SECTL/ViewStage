const TILE_COLS = 4;
const TILE_ROWS = 4;
/** 每帧最多预升级的不可见瓦片数量（分帧降低缩放结束后的单帧卡顿） */
const TILE_UPGRADE_BATCH = 3;

class TileRenderer {
    constructor(options) {
        this.dirty = new Set();
        this.tileInfos = [];
        this._lastDprUpdateScale = 0;
        this._pendingDpr = null;
        this._rebuildRafId = null;
        this._upgradeQueue = null;
        this._upgradeTargetDpr = null;
        this._upgradeRafId = null;
        this._queuedUpgradeKeys = new Set();
        this._quadtree = null;
        this._baseCaches = new Map();
        this._baseCacheLoadId = 0;
        this._dprSettleTimerId = null;
        this._DPR_SETTLE_MS = 300;
        this._idleShrinkTimerId = null;
        this._IDLE_SHRINK_MS = 2000;
        this._strokeVersion = 0;
        this._builtStrokeVersion = -1;
        this._strokeIndex = null;
        this._strokeIndexVersion = -1;

        this._strokeHistoryRef = options?.strokeHistoryRef || null;
        this._getVisibleRectFn = options?.getVisibleRect || null;
        this._canvasW = options?.canvasW || null;
        this._canvasH = options?.canvasH || null;
        this._skipBaseCache = options?.skipBaseCache || false;

        for (let r = 0; r < TILE_ROWS; r++) {
            for (let c = 0; c < TILE_COLS; c++) {
                this.tileInfos.push({ col: c, row: r, key: `${c}_${r}`, dpr: 1 });
            }
        }
        this._init_tile_map();
    }

    _get_stroke_history() {
        return this._strokeHistoryRef || window.state.strokeHistory;
    }

    /**
     * 获取 stroke 按原始顺序的索引 Map，带版本缓存。
     * rebuild_tile 中排序需要按原始顺序，但四叉树不保证顺序，
     * 因此建立 stroke → index 映射。缓存避免每次 tile 重建都遍历全部 strokes。
     */
    _get_or_build_stroke_index() {
        const strokes = this._get_stroke_history();
        const version = this._strokeVersion;
        if (this._strokeIndexVersion !== version) {
            this._strokeIndex = new Map();
            for (let i = 0; i < strokes.length; i++) {
                this._strokeIndex.set(strokes[i], i);
            }
            this._strokeIndexVersion = version;
        }
        return this._strokeIndex;
    }

    _get_canvas_w() {
        return Math.max(1, this._canvasW || window.DRAW_CONFIG.canvasW || 1);
    }

    _get_canvas_h() {
        return Math.max(1, this._canvasH || window.DRAW_CONFIG.canvasH || 1);
    }

    _init_tile_map() {
        this._tileMap = new Map();
        for (const info of this.tileInfos) {
            this._tileMap.set(info.key, info);
        }
    }

    _cancel_dpr_settle() {
        if (this._dprSettleTimerId !== null) {
            clearTimeout(this._dprSettleTimerId);
            this._dprSettleTimerId = null;
        }
    }

    _schedule_dpr_update(scale, force) {
        const targetDpr = this._calc_target_dpr(scale);
        const keys = this.get_visible_keys();
        let changed = false;
        let visibleChanged = false;
        // 检测是否存在：
        //   1) 可见瓦片 DPR 与目标不匹配（需升级或降级）
        //   2) 非可见瓦片 DPR > 1（需缩减至 1 以节约 GPU 内存）
        //   3) 非可见瓦片 DPR 低于目标值（被 idle-shrink 降为 1 后重新进入视野
        //      时需要升级）－ 修复动态分辨率在纯平移后不生效的问题
        for (const info of this.tileInfos) {
            if (keys.has(info.key)) {
                if (info.dpr !== targetDpr) {
                    changed = true;
                    visibleChanged = true;
                    break;
                }
            } else if (info.dpr > 1 || (info.dpr < targetDpr && !this._queuedUpgradeKeys.has(info.key))) {
                changed = true; break;
            }
        }
        if (!changed) return;

        // 非可见瓦片需要升级（dpr < targetDpr）时不应用迟滞，
        // 确保平移后重新进入视野时分辨率正确。迟滞仅用于阻止
        // 缩放边界上的可见瓦片 DPR 反复跳动。
        if (!force && !visibleChanged) {
            let needsUpgrade = false;
            for (const info of this.tileInfos) {
                if (!keys.has(info.key) && info.dpr < targetDpr && !this._queuedUpgradeKeys.has(info.key)) {
                    needsUpgrade = true;
                    break;
                }
            }
            if (!needsUpgrade) {
                const cfg = window.DRAW_CONFIG;
                const hysteresis = Math.max(0.15, (cfg.dprStep || 0.5) / Math.max(0.5, cfg.baseDpr || window.devicePixelRatio || 1));
                if (Math.abs(scale - this._lastDprUpdateScale) < hysteresis) {
                    return;
                }
            }
        }
        this._lastDprUpdateScale = scale;

        this._cancel_pending_rebuild();
        this._pendingDpr = targetDpr;
        this._rebuildRafId = requestAnimationFrame(() => this._apply_dpr_update());
    }

    _update_base_cache() {
        if (this._skipBaseCache) return;
        const img = window.state.baseImageObj;
        const loadId = window.state.baseImageLoadId || 0;
        if (!img) {
            this._clear_base_caches();
            this._baseCacheLoadId = loadId;
            return;
        }
        if (this._baseCacheLoadId === loadId && this._baseCaches.size > 0) return;
        for (const info of this.tileInfos) {
            const rect = info.rect;
            let entry = this._baseCaches.get(info.key);
            if (!entry) {
                const canvas = document.createElement('canvas');
                const ctx = canvas.getContext('2d');
                canvas.width = rect.width;
                canvas.height = rect.height;
                entry = { canvas, ctx };
                this._baseCaches.set(info.key, entry);
            }
            const ctx = entry.ctx;
            ctx.setTransform(1, 0, 0, 1, 0, 0);
            ctx.clearRect(0, 0, rect.width, rect.height);
            ctx.drawImage(img, rect.x, rect.y, rect.width, rect.height, 0, 0, rect.width, rect.height);
        }
        this._baseCacheLoadId = loadId;
    }

    _clear_base_caches() {
        for (const entry of this._baseCaches.values()) {
            entry.canvas = null;
            entry.ctx = null;
        }
        this._baseCaches.clear();
    }

    invalidate_base_cache() {
        this._clear_base_caches();
    }

    mark_strokes_changed() {
        this._strokeVersion++;
    }

    _build_quadtree() {
        if (this._strokeVersion === this._builtStrokeVersion) return;
        this._builtStrokeVersion = this._strokeVersion;
        const strokes = this._get_stroke_history();
        if (!strokes || strokes.length === 0) {
            this._quadtree = null;
            return;
        }
        const boundary = {
            x: 0,
            y: 0,
            width: this._get_canvas_w(),
            height: this._get_canvas_h()
        };
        this._quadtree = new window.StrokeQuadTree(boundary);
        this._quadtree.build(strokes);
    }

    get_tile_dimensions() {
        const cw = Math.max(1, this._get_canvas_w());
        const ch = Math.max(1, this._get_canvas_h());
        return {
            w: Math.max(1, Math.ceil(cw / TILE_COLS)),
            h: Math.max(1, Math.ceil(ch / TILE_ROWS))
        };
    }

    get_tile_rect(col, row) {
        const { w, h } = this.get_tile_dimensions();
        const cw = this._get_canvas_w();
        const ch = this._get_canvas_h();
        return {
            x: col * w,
            y: row * h,
            width: Math.min(w, cw - col * w),
            height: Math.min(h, ch - row * h)
        };
    }

    tile_key(col, row) { return `${col}_${row}`; }

    _calc_target_dpr(scale) {
        const cfg = window.DRAW_CONFIG;
        if (cfg.dynamicDprEnabled === false) return cfg.dpr;
        const baseDpr = cfg.baseDpr || window.devicePixelRatio || 1;
        const minDpr = cfg.dprMin || 1;
        const maxDpr = cfg.dprMax || 4;
        const step = cfg.dprStep || 0.25;
        // 改用 Math.ceil 使 DPR 始终不低于原始计算值，避免向下取整导致的模糊
        let dpr = baseDpr * scale;
        dpr = Math.ceil(dpr / step) * step;
        return Math.max(minDpr, Math.min(maxDpr, dpr));
    }

    _create_tile_canvas(info, dpr) {
        const rect = info.rect;
        const canvas = document.createElement('canvas');
        canvas.className = 'canvas-tile';
        canvas.width = Math.ceil(rect.width * dpr);
        canvas.height = Math.ceil(rect.height * dpr);
        canvas.style.width = rect.width + 'px';
        canvas.style.height = rect.height + 'px';
        canvas.style.left = rect.x + 'px';
        canvas.style.top = rect.y + 'px';

        const ctx = canvas.getContext('2d', { alpha: true });
        ctx.imageSmoothingEnabled = false;
        ctx.lineCap = 'round';
        ctx.lineJoin = 'round';

        return { canvas, ctx };
    }

    _recreate_tile(info, newDpr) {
        const canvas = info.canvas;
        const ctx = info.ctx;
        if (!canvas || !ctx) return;

        // 仅在 DPR 提升时保存 snapshot（缩小后必然被 rebuild 覆盖，无需占位）
        let snapshot = null;
        if (newDpr > info.dpr && canvas.width > 0 && canvas.height > 0) {
            snapshot = document.createElement('canvas');
            snapshot.width = canvas.width;
            snapshot.height = canvas.height;
            snapshot.getContext('2d').drawImage(canvas, 0, 0);
        }

        canvas.width = Math.ceil(info.rect.width * newDpr);
        canvas.height = Math.ceil(info.rect.height * newDpr);
        ctx.imageSmoothingEnabled = false;
        ctx.lineCap = 'round';
        ctx.lineJoin = 'round';

        // 写入旧内容（scaled）作为占位图，rebuild_tile 会覆盖精确内容
        if (snapshot) {
            ctx.save();
            ctx.imageSmoothingEnabled = true;
            ctx.drawImage(snapshot, 0, 0, canvas.width, canvas.height);
            ctx.restore();
            ctx.imageSmoothingEnabled = false;
        }

        info.dpr = newDpr;
        this.dirty.add(info.key);
    }

    update_visible_tile_dpr(scale, force, skipSettle) {
        if (skipSettle) {
            this._cancel_dpr_settle();
            this._schedule_dpr_update(scale, force);
            return;
        }
        this._cancel_dpr_settle();
        this._dprSettleTimerId = setTimeout(() => {
            this._dprSettleTimerId = null;
            this._schedule_dpr_update(scale, force);
        }, this._DPR_SETTLE_MS);
    }

    cancel_idle_shrink() {
        this._cancel_idle_shrink();
    }

    _cancel_pending_rebuild() {
        this._cancel_dpr_settle();
        if (this._rebuildRafId !== null) {
            cancelAnimationFrame(this._rebuildRafId);
            this._rebuildRafId = null;
        }
        this._pendingDpr = null;
        this._cancel_upgrade_queue();
    }

    /** 取消未完成的不可见瓦片预升级队列 */
    _cancel_upgrade_queue() {
        if (this._upgradeRafId !== null) {
            cancelAnimationFrame(this._upgradeRafId);
            this._upgradeRafId = null;
        }
        this._upgradeQueue = null;
        this._upgradeTargetDpr = null;
        this._queuedUpgradeKeys.clear();
    }

    /** 分帧消费预升级队列，每帧最多 TILE_UPGRADE_BATCH 块 */
    _pump_upgrade_queue() {
        this._upgradeRafId = null;
        const targetDpr = this._upgradeTargetDpr;
        if (!this._upgradeQueue || this._upgradeQueue.length === 0 || targetDpr == null) {
            this._cancel_upgrade_queue();
            return;
        }
        let budget = TILE_UPGRADE_BATCH;
        while (budget-- > 0 && this._upgradeQueue.length > 0) {
            const info = this._upgradeQueue.shift();
            this._queuedUpgradeKeys.delete(info.key);
            // 目标期间可能被 idle-shrink 降低过；仅升级仍低于目标的瓦片
            if (info.dpr < targetDpr) {
                this._recreate_tile(info, targetDpr);
            }
        }
        if (this._upgradeQueue.length > 0) {
            this._upgradeRafId = requestAnimationFrame(() => this._pump_upgrade_queue());
        } else {
            this._cancel_upgrade_queue();
        }
    }

    _cancel_idle_shrink() {
        if (this._idleShrinkTimerId != null) {
            clearTimeout(this._idleShrinkTimerId);
            this._idleShrinkTimerId = null;
        }
    }

    _schedule_idle_shrink() {
        this._cancel_idle_shrink();
        this._idleShrinkTimerId = setTimeout(() => {
            this._idleShrinkTimerId = null;
            const keys = this.get_visible_keys();
            let anyShrunk = false;
            for (const info of this.tileInfos) {
                if (!keys.has(info.key) && info.dpr > 1) {
                    this._recreate_tile(info, 1);
                    anyShrunk = true;
                }
            }
            if (anyShrunk) {
                this.rebuild_visible(keys);
            }
        }, this._IDLE_SHRINK_MS);
    }

    _apply_dpr_update() {
        this._rebuildRafId = null;
        const targetDpr = this._pendingDpr;
        this._pendingDpr = null;
        if (targetDpr == null) return;

        // 中断上一轮未完成的预升级队列（目标 DPR 可能已变化）
        this._cancel_upgrade_queue();

        const keys = this.get_visible_keys();
        const deferredUpgrades = [];
        for (const info of this.tileInfos) {
            if (keys.has(info.key)) {
                if (info.dpr !== targetDpr) {
                    this._recreate_tile(info, targetDpr);
                }
            } else if (info.dpr < targetDpr) {
                // 预升级非可见瓦片，使其在进入视野时已有正确分辨率。
                // 不可见块延后分帧处理：一次 realloc 大量画布会造成缩放收尾掉帧
                deferredUpgrades.push(info);
            }
        }
        this.rebuild_visible(keys);
        this._schedule_idle_shrink();

        if (deferredUpgrades.length > 0) {
            this._upgradeQueue = deferredUpgrades;
            this._upgradeTargetDpr = targetDpr;
            for (const info of deferredUpgrades) {
                this._queuedUpgradeKeys.add(info.key);
            }
            this._upgradeRafId = requestAnimationFrame(() => this._pump_upgrade_queue());
        }
    }

    init_tiles(wrapper, initialScale) {
        const scale = initialScale || (window.state ? (window.state.scale || 1) : 1);
        const existing = wrapper.querySelectorAll('.canvas-tile');
        for (const el of existing) el.remove();

        for (const info of this.tileInfos) {
            info.rect = this.get_tile_rect(info.col, info.row);
            const dpr = this._calc_target_dpr(scale);
            const { canvas, ctx } = this._create_tile_canvas(info, dpr);

            wrapper.appendChild(canvas);
            info.canvas = canvas;
            info.ctx = ctx;
            info.dpr = dpr;
            this.dirty.add(info.key);
        }
    }

    for_each_visible(fn) {
        const keys = this.get_visible_keys();
        for (const info of this.tileInfos) {
            if (keys.has(info.key)) {
                fn(info);
            }
        }
    }

    for_each(fn) {
        for (const info of this.tileInfos) {
            fn(info);
        }
    }

    get_visible_keys() {
        const vr = this._getVisibleRectFn ? this._getVisibleRectFn() : window.main_fetch_visible_rect();
        const { w, h } = this.get_tile_dimensions();
        const keys = new Set();
        const sc = Math.max(0, Math.floor(vr.x / w));
        const ec = Math.min(TILE_COLS - 1, Math.floor((vr.x + vr.width - 1) / w));
        const sr = Math.max(0, Math.floor(vr.y / h));
        const er = Math.min(TILE_ROWS - 1, Math.floor((vr.y + vr.height - 1) / h));
        for (let r = sr; r <= er; r++) {
            for (let c = sc; c <= ec; c++) {
                keys.add(this.tile_key(c, r));
            }
        }
        return keys;
    }

    info_for_point(x, y) {
        const { w, h } = this.get_tile_dimensions();
        const col = Math.min(TILE_COLS - 1, Math.max(0, Math.floor(x / w)));
        const row = Math.min(TILE_ROWS - 1, Math.max(0, Math.floor(y / h)));
        return this._tileMap.get(this.tile_key(col, row));
    }

    infos_for_segment(x1, y1, x2, y2, padding = 0) {
        const { w, h } = this.get_tile_dimensions();
        const minX = Math.min(x1, x2) - padding;
        const maxX = Math.max(x1, x2) + padding;
        const minY = Math.min(y1, y2) - padding;
        const maxY = Math.max(y1, y2) + padding;
        const sc = Math.max(0, Math.floor(minX / w));
        const ec = Math.min(TILE_COLS - 1, Math.floor(maxX / w));
        const sr = Math.max(0, Math.floor(minY / h));
        const er = Math.min(TILE_ROWS - 1, Math.floor(maxY / h));
        const result = [];
        for (let r = sr; r <= er; r++) {
            for (let c = sc; c <= ec; c++) {
                const info = this._tileMap.get(this.tile_key(c, r));
                if (info) result.push(info);
            }
        }
        return result;
    }

    rebuild_tile(info) {
        const ctx = info.ctx;
        const rect = info.rect;
        const dpr = info.dpr;
        ctx.save();
        ctx.setTransform(dpr, 0, 0, dpr, -rect.x * dpr, -rect.y * dpr);
        ctx.clearRect(rect.x, rect.y, rect.width, rect.height);

        const cacheEntry = this._baseCaches.get(info.key);
        if (cacheEntry) {
            ctx.drawImage(
                cacheEntry.canvas,
                0, 0, rect.width, rect.height,
                rect.x, rect.y, rect.width, rect.height
            );
        }

        const strokes = this._get_stroke_history();
        if (strokes.length > 0) {
            let relevant;
            if (this._quadtree) {
                relevant = Array.from(this._quadtree.query({
                    x: rect.x, y: rect.y,
                    width: rect.width, height: rect.height
                }));
                // 四叉树 Set 不保证插入顺序，必须按原始 strokes 顺序排序
                // 确保橡皮擦 destination-out 在绘制 stroke 之后执行
                if (relevant.length > 1) {
                    const stroke_index = this._get_or_build_stroke_index();
                    relevant.sort((a, b) => stroke_index.get(a) - stroke_index.get(b));
                }
            } else {
                relevant = [];
                for (let i = 0; i < strokes.length; i++) {
                    const s = strokes[i];
                    if (!s.bounds) { relevant.push(s); continue; }
                    const b = s.bounds;
                    if (b.maxX < rect.x || b.minX > rect.x + rect.width ||
                        b.maxY < rect.y || b.minY > rect.y + rect.height) {
                        continue;
                    }
                    relevant.push(s);
                }
            }
            if (relevant.length > 0) {
                ctx.save();
                ctx.beginPath();
                ctx.rect(rect.x, rect.y, rect.width, rect.height);
                ctx.clip();
                if (window.main_reset_context_state) window.main_reset_context_state();
                window.main_render_strokes_to_context(ctx, relevant);
                ctx.restore();
            }
        }

        ctx.restore();
        this.dirty.delete(info.key);
    }

    rebuild_visible(keys) {
        if (!keys) keys = this.get_visible_keys();
        this._build_quadtree();
        this._update_base_cache();
        for (const info of this.tileInfos) {
            if (keys.has(info.key) && this.dirty.has(info.key)) {
                this.rebuild_tile(info);
            }
        }
    }

    rebuild_all() {
        this._build_quadtree();
        this._update_base_cache();
        for (const info of this.tileInfos) {
            if (this.dirty.has(info.key)) {
                this.rebuild_tile(info);
            }
        }
    }

    mark_all() {
        for (const info of this.tileInfos) {
            this.dirty.add(info.key);
        }
    }

    mark_visible() {
        const keys = this.get_visible_keys();
        for (const info of this.tileInfos) {
            if (keys.has(info.key)) {
                this.dirty.add(info.key);
            }
        }
    }

    destroy() {
        this._cancel_pending_rebuild();
        this._cancel_dpr_settle();
        this._cancel_idle_shrink();
        for (const info of this.tileInfos) {
            if (info.canvas && info.canvas.parentNode) {
                info.canvas.parentNode.removeChild(info.canvas);
            }
            info.canvas = null;
            info.ctx = null;
        }
        this._clear_base_caches();
        this._baseCacheLoadId = 0;
        this._strokeIndex = null;
        this._strokeIndexVersion = -1;
        this.dirty.clear();
    }

    destroy_all() {
        this.destroy();
    }

    add_stroke(stroke) {
        if (!stroke || !stroke.points || stroke.points.length < 2) return;
        if (this._quadtree) {
            this._quadtree.insert(stroke);
        }
        const halfWidth = Math.max(stroke.lineWidth || 5, stroke.eraserSize || 5) / 2;
        const infos = this.infos_for_segment(
            stroke.bounds.minX, stroke.bounds.minY,
            stroke.bounds.maxX, stroke.bounds.maxY,
            halfWidth
        );
        if (infos.length === 0) return;

        const uniqueKeys = new Set(infos.map(i => i.key));

        /* 单 tile —— 直接渲染，无需 offscreen 中间层 */
        if (infos.length === 1) {
            const info = infos[0];
            const ctx = info.ctx;
            const rect = info.rect;
            const dpr = info.dpr;
            ctx.save();
            ctx.setTransform(dpr, 0, 0, dpr, -rect.x * dpr, -rect.y * dpr);
            window.main_render_strokes_to_context(ctx, [stroke]);
            ctx.restore();
            this.dirty.delete(info.key);
            return;
        }

        /* 多 tile —— 同 DPR 前提下用 offscreen canvas 预渲染后 blit */
        /* 注：不存在多显示器混合 DPR 场景，无需对 DPR 分组 */
        const mainDpr = infos[0].dpr;
        if (infos.some(i => i.dpr !== mainDpr)) {
            for (const info of this.tileInfos) {
                if (!uniqueKeys.has(info.key)) continue;
                const ctx = info.ctx;
                const rect = info.rect;
                const dpr = info.dpr;
                ctx.save();
                ctx.setTransform(dpr, 0, 0, dpr, -rect.x * dpr, -rect.y * dpr);
                window.main_render_strokes_to_context(ctx, [stroke]);
                ctx.restore();
                this.dirty.delete(info.key);
            }
            return;
        }

        /* 多 tile 同 DPR —— 预渲染到 offscreen canvas，drawImage 复合各 tile */
        const sx = stroke.bounds.minX - halfWidth;
        const sy = stroke.bounds.minY - halfWidth;
        const sw = stroke.bounds.maxX - stroke.bounds.minX + halfWidth * 2;
        const sh = stroke.bounds.maxY - stroke.bounds.minY + halfWidth * 2;

        const offscreen = document.createElement('canvas');
        offscreen.width = Math.ceil(sw * mainDpr);
        offscreen.height = Math.ceil(sh * mainDpr);
        const offCtx = offscreen.getContext('2d');
        offCtx.setTransform(mainDpr, 0, 0, mainDpr, -sx * mainDpr, -sy * mainDpr);
        window.main_render_strokes_to_context(offCtx, [stroke]);

        /* 将预渲染结果 blit 到各 tile，仅绘制 tile 与 stroke 交集区域 */
        for (const info of this.tileInfos) {
            if (!uniqueKeys.has(info.key)) continue;
            const ctx = info.ctx;
            const rect = info.rect;
            const clipX = Math.max(rect.x, sx);
            const clipY = Math.max(rect.y, sy);
            const clipR = Math.min(rect.x + rect.width, sx + sw);
            const clipB = Math.min(rect.y + rect.height, sy + sh);
            const clipW = clipR - clipX;
            const clipH = clipB - clipY;
            if (clipW <= 0 || clipH <= 0) continue;

            ctx.save();
            ctx.setTransform(mainDpr, 0, 0, mainDpr, -rect.x * mainDpr, -rect.y * mainDpr);
            ctx.drawImage(
                offscreen,
                (clipX - sx) * mainDpr, (clipY - sy) * mainDpr,
                clipW * mainDpr, clipH * mainDpr,
                clipX, clipY, clipW, clipH
            );
            ctx.restore();
            this.dirty.delete(info.key);
        }
    }
}

window.TileRenderer = TileRenderer;
window.tileRenderer = new TileRenderer();
