
import React, { useState, useRef, useEffect, useMemo, useCallback } from 'react';
import ReactDOM from 'react-dom/client';
import { GoogleGenAI } from "@google/genai";

const API_KEY = process.env.API_KEY;
const MAX_HISTORY = 30; // Increased for more complex workflows

// --- Types & Initial State ---
type BasicFilters = {
  brightness: number;
  contrast: number;
  saturation: number;
  sepia: number;
  exposure: number;
  highlights: number;
  shadows: number;
  whites: number;
  blacks: number;
  temperature: number;
  vibrance: number;
  sharpen: number;
  dehaze: number;
  grain: number;
  haze: number;
  hazeSpread: number;
};

type HSLColor = { h: number; s: number; l: number };
type HSLFilters = {
    red: HSLColor;
    orange: HSLColor;
    yellow: HSLColor;
    green: HSLColor;
    aqua: HSLColor;
    blue: HSLColor;
    purple: HSLColor;
    magenta: HSLColor;
};

type Curve = { x: number; y: number }[];
type CurvesState = {
    rgb: Curve;
    red: Curve;
    green: Curve;
    blue: Curve;
};

type Adjustments = {
    filters: BasicFilters;
    hsl: HSLFilters;
    curves: CurvesState;
};

type Layer = {
    id: string;
    name: string;
    isVisible: boolean;
    maskSrc: string | null;
    isMaskInverted: boolean;
};

type EditorState = {
    baseImageSrc: string | null;
    layers: Layer[];
    adjustments: Adjustments;
};

type HistoryState = {
    stack: EditorState[];
    index: number;
}

type HistogramData = {
    rgb: number[];
    red: number[];
    green: number[];
    blue: number[];
};

type Theme = 'light' | 'dark' | 'mono-light' | 'mono-dark';
type CustomPreset = { name: string; adjustments: Adjustments };
type DownloadOptions = { format: 'image/png' | 'image/jpeg', quality: number };
type Transform = { zoom: number; pan: { x: number; y: number } };
type BrushMode = 'none' | 'mask' | 'eraser' | 'sketch' | 'magic-remove' | 'linear-gradient' | 'radial-gradient';
type Point = { x: number, y: number };
type CropBox = { x: number; y: number; width: number; height: number };

// --- Image Processing Helpers ---

function rgbToHsl(r: number, g: number, b: number): [number, number, number] {
    r /= 255; g /= 255; b /= 255;
    const max = Math.max(r, g, b), min = Math.min(r, g, b);
    let h = 0, s = 0, l = (max + min) / 2;
    if (max !== min) {
        const d = max - min;
        s = l > 0.5 ? d / (2 - max - min) : d / (max + min);
        switch (max) {
            case r: h = (g - b) / d + (g < b ? 6 : 0); break;
            case g: h = (b - r) / d + 2; break;
            case b: h = (r - g) / d + 4; break;
        }
        h /= 6;
    }
    return [h * 360, s, l];
}

function hslToRgb(h: number, s: number, l: number): [number, number, number] {
    let r, g, b;
    h /= 360;
    if (s === 0) {
        r = g = b = l;
    } else {
        const hue2rgb = (p: number, q: number, t: number) => {
            if (t < 0) t += 1;
            if (t > 1) t -= 1;
            if (t < 1 / 6) return p + (q - p) * 6 * t;
            if (t < 1 / 2) return q;
            if (t < 2 / 3) return p + (q - p) * (2 / 3 - t) * 6;
            return p;
        };
        const q = l < 0.5 ? l * (1 + s) : l + s - l * s;
        const p = 2 * l - q;
        r = hue2rgb(p, q, h + 1 / 3);
        g = hue2rgb(p, q, h);
        b = hue2rgb(p, q, h - 1 / 3);
    }
    return [r * 255, g * 255, b * 255];
}

const HSL_RANGES = {
    red:    { center: 0,   range: 90 },
    orange: { center: 30,  range: 90 },
    yellow: { center: 60,  range: 90 },
    green:  { center: 120, range: 150 },
    aqua:   { center: 180, range: 90 },
    blue:   { center: 240, range: 150 },
    purple: { center: 285, range: 120 },
    magenta:{ center: 330, range: 90 }
};

function applyHsl(imageData: ImageData, hslAdjustments: HSLFilters): ImageData {
    const data = imageData.data;
    for (let i = 0; i < data.length; i += 4) {
        const [r, g, b] = [data[i], data[i + 1], data[i + 2]];
        let [h, s, l] = rgbToHsl(r, g, b);

        let totalHueChange = 0;
        let totalSatChange = 0;
        let totalLumChange = 0;
        let totalInfluence = 0;

        for (const [color, { center, range }] of Object.entries(HSL_RANGES)) {
            const adj = hslAdjustments[color as keyof HSLFilters];
            if (adj.h === 0 && adj.s === 0 && adj.l === 0) continue;
            
            const dist = Math.min(Math.abs(h - center), 360 - Math.abs(h - center));
            
            let influence = 0;
            if (dist < range / 2) {
                influence = Math.pow((Math.cos((dist / (range / 2)) * Math.PI) + 1) / 2, 2);
            }

            if (influence > 0) {
                totalHueChange += (adj.h / 100) * 180 * influence;
                totalSatChange += (adj.s / 100) * influence;
                totalLumChange += (adj.l / 100) * influence;
                totalInfluence += influence;
            }
        }
        
        if (totalInfluence > 1) {
            totalHueChange /= totalInfluence;
            totalSatChange /= totalInfluence;
            totalLumChange /= totalInfluence;
        }

        h = (h + totalHueChange + 360) % 360;
        s = Math.max(0, Math.min(1, s + totalSatChange));
        l = Math.max(0, Math.min(1, l + totalLumChange));

        const [newR, newG, newB] = hslToRgb(h, s, l);
        data[i] = newR;
        data[i + 1] = newG;
        data[i + 2] = newB;
    }
    return imageData;
}

function applyHighlightsShadows(imageData: ImageData, highlights: number, shadows: number): ImageData {
    const data = imageData.data;
    const h_adj = highlights / 100.0; // -1 to 1
    const s_adj = shadows / 100.0;   // -1 to 1

    if (h_adj === 0 && s_adj === 0) return imageData;

    for (let i = 0; i < data.length; i += 4) {
        let [h, s, l] = rgbToHsl(data[i], data[i + 1], data[i + 2]);
        const shadow_mask = l < 0.5 ? Math.pow(Math.cos(l * Math.PI), 2) : 0;
        const shadow_boost = s_adj * shadow_mask;
        const highlight_mask = l > 0.5 ? Math.pow(Math.cos((1 - l) * Math.PI), 2) : 0;
        const highlight_boost = h_adj * highlight_mask;
        
        let new_l = l + shadow_boost + highlight_boost;
        new_l = Math.max(0, Math.min(1, new_l));

        const [newR, newG, newB] = hslToRgb(h, s, new_l);
        data[i] = newR;
        data[i + 1] = newG;
        data[i + 2] = newB;
    }
    return imageData;
}

function applyWhitesBlacks(imageData: ImageData, whites: number, blacks: number): ImageData {
    if (whites === 0 && blacks === 0) return imageData;
    const data = imageData.data;
    const whites_adj = whites / 100.0;
    const blacks_adj = blacks / 100.0;

    for (let i = 0; i < data.length; i += 4) {
        let r = data[i];
        let g = data[i + 1];
        let b = data[i + 2];
        const luma = (r * 0.299 + g * 0.587 + b * 0.114) / 255.0;

        if (whites_adj !== 0) {
            const white_factor = whites_adj * luma * luma;
            r = Math.max(0, Math.min(255, r + white_factor * 255));
            g = Math.max(0, Math.min(255, g + white_factor * 255));
            b = Math.max(0, Math.min(255, b + white_factor * 255));
        }
        
        if (blacks_adj !== 0) {
            const black_factor = blacks_adj * (1.0 - luma) * (1.0 - luma);
            r = Math.max(0, Math.min(255, r - black_factor * 255));
            g = Math.max(0, Math.min(255, g - black_factor * 255));
            b = Math.max(0, Math.min(255, b - black_factor * 255));
        }
        
        data[i] = r;
        data[i+1] = g;
        data[i+2] = b;
    }
    return imageData;
}

function applyExposure(imageData: ImageData, amount: number): ImageData {
    if (amount === 0) return imageData;
    const data = imageData.data;
    const multiplier = Math.pow(2, amount / 100.0);
    for (let i = 0; i < data.length; i += 4) {
        data[i] = Math.max(0, Math.min(255, data[i] * multiplier));
        data[i + 1] = Math.max(0, Math.min(255, data[i + 1] * multiplier));
        data[i + 2] = Math.max(0, Math.min(255, data[i + 2] * multiplier));
    }
    return imageData;
}

function applyTemperature(imageData: ImageData, temperature: number): ImageData {
    const amount = temperature / 2.0;
    if (amount === 0) return imageData;
    const data = imageData.data;
    for (let i = 0; i < data.length; i += 4) {
        data[i] = Math.max(0, Math.min(255, data[i] + amount));
        data[i+2] = Math.max(0, Math.min(255, data[i+2] - amount));
    }
    return imageData;
}

function applySharpen(imageData: ImageData, amount: number): ImageData {
    const strength = amount / 100;
    if (strength === 0) return imageData;
    const w = imageData.width;
    const h = imageData.height;
    const src = new Uint8ClampedArray(imageData.data);
    const dst = imageData.data;
    
    const kernel = [
        0, -1 * strength, 0,
        -1 * strength, 1 + 4 * strength, -1 * strength,
        0, -1 * strength, 0
    ];

    for (let y = 1; y < h - 1; y++) {
        for (let x = 1; x < w - 1; x++) {
            for (let c = 0; c < 3; c++) {
                const i = (y * w + x) * 4 + c;
                let total = 0;
                for (let ky = -1; ky <= 1; ky++) {
                    for (let kx = -1; kx <= 1; kx++) {
                        const sy = y + ky;
                        const sx = x + kx;
                        const si = (sy * w + sx) * 4 + c;
                        const ki = (ky + 1) * 3 + (kx + 1);
                        total += src[si] * kernel[ki];
                    }
                }
                dst[i] = Math.max(0, Math.min(255, total));
            }
        }
    }
    return imageData;
}

function applyGrain(imageData: ImageData, amount: number): ImageData {
    if (amount === 0) return imageData;
    const data = imageData.data;
    const strength = amount * 2.55;
    for (let i = 0; i < data.length; i += 4) {
        const noise = (Math.random() - 0.5) * strength;
        data[i] = Math.max(0, Math.min(255, data[i] + noise));
        data[i + 1] = Math.max(0, Math.min(255, data[i + 1] + noise));
        data[i + 2] = Math.max(0, Math.min(255, data[i + 2] + noise));
    }
    return imageData;
}

function applyVibrance(imageData: ImageData, amount: number): ImageData {
    const data = imageData.data;
    const adjust = amount * 1.5;
    if (adjust === 0) return imageData;
    for (let i = 0; i < data.length; i += 4) {
        const r = data[i], g = data[i + 1], b = data[i + 2];
        const max = Math.max(r, g, b);
        const avg = (r + g + b) / 3;
        const sat = Math.abs(max - avg);
        const boost = (adjust / 255) * (1 - sat / 128);
        if (boost <= 0 && amount > 0) continue;
        if (boost >=0 && amount < 0) continue;

        data[i] = Math.max(0, Math.min(255, r + (max - r) * boost));
        data[i+1] = Math.max(0, Math.min(255, g + (max - g) * boost));
        data[i+2] = Math.max(0, Math.min(255, b + (max - b) * boost));
    }
    return imageData;
}

function applyDehaze(imageData: ImageData, amount: number): ImageData {
    if (amount === 0) return imageData;
    const data = imageData.data;
    const strength = amount / 100.0;
    const airlight = { r: 220, g: 220, b: 230 };
    const t0 = 0.1;

    for (let i = 0; i < data.length; i+=4) {
        const transmission = 1.0 - strength * Math.min(data[i]/airlight.r, data[i+1]/airlight.g, data[i+2]/airlight.b);
        data[i] = Math.max(0, Math.min(255, (data[i] - airlight.r) / Math.max(transmission, t0) + airlight.r));
        data[i+1] = Math.max(0, Math.min(255, (data[i+1] - airlight.g) / Math.max(transmission, t0) + airlight.g));
        data[i+2] = Math.max(0, Math.min(255, (data[i+2] - airlight.b) / Math.max(transmission, t0) + airlight.b));
    }
    return imageData;
}

function boxBlur(src: ImageData, dst: ImageData, w: number, h: number, r: number) {
    const srcData = src.data;
    const dstData = dst.data;
    const tempSrc = new Uint8ClampedArray(srcData.length);
    tempSrc.set(srcData);

    // Horizontal pass
    for (let y = 0; y < h; y++) {
        for (let x = 0; x < w; x++) {
            let r_acc = 0, g_acc = 0, b_acc = 0, a_acc = 0;
            let count = 0;
            for (let i = -r; i <= r; i++) {
                const xi = x + i;
                if (xi >= 0 && xi < w) {
                    const src_idx = (y * w + xi) * 4;
                    r_acc += tempSrc[src_idx];
                    g_acc += tempSrc[src_idx + 1];
                    b_acc += tempSrc[src_idx + 2];
                    a_acc += tempSrc[src_idx + 3];
                    count++;
                }
            }
            const dst_idx = (y * w + x) * 4;
            dstData[dst_idx]     = r_acc / count;
            dstData[dst_idx + 1] = g_acc / count;
            dstData[dst_idx + 2] = b_acc / count;
            dstData[dst_idx + 3] = a_acc / count;
        }
    }
    
    tempSrc.set(dstData); // Use the horizontally blurred result for the vertical pass

    // Vertical pass
    for (let x = 0; x < w; x++) {
        for (let y = 0; y < h; y++) {
            let r_acc = 0, g_acc = 0, b_acc = 0, a_acc = 0;
            let count = 0;
            for (let i = -r; i <= r; i++) {
                const yi = y + i;
                if (yi >= 0 && yi < h) {
                    const src_idx = (yi * w + x) * 4;
                    r_acc += tempSrc[src_idx];
                    g_acc += tempSrc[src_idx + 1];
                    b_acc += tempSrc[src_idx + 2];
                    a_acc += tempSrc[src_idx + 3];
                    count++;
                }
            }
            const dst_idx = (y * w + x) * 4;
            dstData[dst_idx]     = r_acc / count;
            dstData[dst_idx + 1] = g_acc / count;
            dstData[dst_idx + 2] = b_acc / count;
            dstData[dst_idx + 3] = a_acc / count;
        }
    }
}

function applyHaze(imageData: ImageData, amount: number, spread: number): ImageData {
    if (amount === 0) return imageData;
    
    const w = imageData.width;
    const h = imageData.height;
    const data = imageData.data;

    // 1. Calculate Haze Color from 'spread' (Warmth)
    const warmth = (spread - 50) / 50.0;
    let hazeR = 230, hazeG = 230, hazeB = 230;
    if (warmth > 0) {
        hazeR = Math.min(255, hazeR + 25 * warmth);
        hazeG = Math.min(255, hazeG + 10 * warmth);
        hazeB = Math.max(0, hazeB - 25 * warmth);
    } else {
        hazeR = Math.max(0, hazeR + 25 * warmth);
        hazeG = Math.max(0, hazeG + 5 * warmth);
        hazeB = Math.min(255, hazeB - 40 * warmth);
    }

    // 2. Create a "glow map" based on luminance
    const glowMap = new ImageData(w, h);
    for (let i = 0; i < data.length; i += 4) {
        const r = data[i], g = data[i + 1], b = data[i + 2];
        const luma = (r * 0.299 + g * 0.587 + b * 0.114);
        glowMap.data[i] = hazeR;
        glowMap.data[i + 1] = hazeG;
        glowMap.data[i + 2] = hazeB;
        glowMap.data[i + 3] = luma * luma / 255; // Use squared luma for stronger highlights, then normalize
    }

    // 3. Blur the glow map
    const blurRadius = Math.floor(Math.max(1, (amount / 100) * (Math.min(w, h) * 0.05)));
    const blurredGlowMap = new ImageData(w, h);
    boxBlur(glowMap, blurredGlowMap, w, h, blurRadius);

    // 4. Composite blurred glow map onto original image
    const hazeAmount = amount / 100.0;
    for (let i = 0; i < data.length; i += 4) {
        const glowAlpha = blurredGlowMap.data[i + 3] / 255.0;
        const blend = glowAlpha * hazeAmount;
        data[i]     = data[i] * (1 - blend) + blurredGlowMap.data[i] * blend;
        data[i + 1] = data[i + 1] * (1 - blend) + blurredGlowMap.data[i + 1] * blend;
        data[i + 2] = data[i + 2] * (1 - blend) + blurredGlowMap.data[i + 2] * blend;
    }

    return imageData;
}


// --- Curves Helpers ---

function createLutFromPoints(points: Curve): number[] {
    const pointMap = new Map<number, number>();
    points.forEach(p => pointMap.set(p.x, p.y));
    const uniquePoints = Array.from(pointMap.entries()).map(([x, y]) => ({ x, y }))
                              .sort((a, b) => a.x - b.x);

    if (uniquePoints.length === 0) {
        return Array.from({ length: 256 }, (_, i) => i);
    }
    if (uniquePoints.length === 1) {
        const yVal = Math.max(0, Math.min(255, uniquePoints[0].y));
        return new Array(256).fill(yVal);
    }

    const n = uniquePoints.length;
    const x = uniquePoints.map(p => p.x);
    const y = uniquePoints.map(p => p.y);
    const m = new Array(n).fill(0);
    const dx = new Array(n - 1).fill(0);
    const delta = new Array(n - 1).fill(0);

    for (let i = 0; i < n - 1; i++) {
        dx[i] = x[i + 1] - x[i];
        if (dx[i] <= 1e-7) {
             delta[i] = 0;
        } else {
             delta[i] = (y[i + 1] - y[i]) / dx[i];
        }
    }

    m[0] = delta[0];
    m[n - 1] = delta[n - 2];
    for (let i = 1; i < n - 1; i++) {
        m[i] = (delta[i - 1] + delta[i]) / 2;
    }

    for (let i = 0; i < n - 1; i++) {
        if (delta[i] === 0) {
            m[i] = 0;
            m[i + 1] = 0;
        } else {
            const alpha = m[i] / delta[i];
            const beta = m[i + 1] / delta[i];
            if (alpha < 0.0) m[i] = 0;
            if (beta < 0.0) m[i + 1] = 0;
            const hyp = Math.sqrt(alpha * alpha + beta * beta);
            if (hyp > 3.0) {
                const tau = 3.0 / hyp;
                m[i] *= tau;
                m[i + 1] *= tau;
            }
        }
    }
    
    const lut = new Array(256);
    let pointIndex = 0;
    for (let i = 0; i < 256; i++) {
        while (pointIndex < n - 2 && i > x[pointIndex + 1]) {
            pointIndex++;
        }
        const h = dx[pointIndex];
        const t = (h > 1e-7) ? (i - x[pointIndex]) / h : 0;
        const y0 = y[pointIndex];
        const y1 = y[pointIndex + 1];
        const m0 = m[pointIndex] * h;
        const m1 = m[pointIndex + 1] * h;
        const t2 = t * t;
        const t3 = t2 * t;
        const h00 = 2 * t3 - 3 * t2 + 1;
        const h10 = t3 - 2 * t2 + t;
        const h01 = -2 * t3 + 3 * t2;
        const h11 = t3 - t2;
        const val = h00 * y0 + h10 * m0 + h01 * y1 + h11 * m1;
        lut[i] = Math.max(0, Math.min(255, val));
    }
    lut[255] = y[n - 1];

    return lut;
}

const BAYER_MATRIX_8X8 = [
  [  0, 32,  8, 40,  2, 34, 10, 42 ],
  [ 48, 16, 56, 24, 50, 18, 58, 26 ],
  [ 12, 44,  4, 36, 14, 46,  6, 38 ],
  [ 60, 28, 52, 20, 62, 30, 54, 22 ],
  [  3, 35, 11, 43,  1, 33,  9, 41 ],
  [ 51, 19, 59, 27, 49, 17, 57, 25 ],
  [ 15, 47,  7, 39, 13, 45,  5, 37 ],
  [ 63, 31, 55, 23, 61, 29, 53, 21 ]
];

function applyCurves(imageData: ImageData, curves: CurvesState): ImageData {
    const data = imageData.data;
    const { width } = imageData;
    const lutRgb = createLutFromPoints(curves.rgb);
    const lutR = createLutFromPoints(curves.red);
    const lutG = createLutFromPoints(curves.green);
    const lutB = createLutFromPoints(curves.blue);

    const isRedDefault = curves.red.length === 2 && curves.red[0].x === 0 && curves.red[0].y === 0 && curves.red[1].x === 255 && curves.red[1].y === 255;
    const isGreenDefault = curves.green.length === 2 && curves.green[0].x === 0 && curves.green[0].y === 0 && curves.green[1].x === 255 && curves.green[1].y === 255;
    const isBlueDefault = curves.blue.length === 2 && curves.blue[0].x === 0 && curves.blue[0].y === 0 && curves.blue[1].x === 255 && curves.blue[1].y === 255;
    
    const needsHueMasking = !isRedDefault || !isGreenDefault || !isBlueDefault;

    for (let i = 0; i < data.length; i += 4) {
        const pixelIndex = i / 4;
        const x = pixelIndex % width;
        const y = Math.floor(pixelIndex / width);
        const dither = (BAYER_MATRIX_8X8[y % 8][x % 8] / 64.0) - 0.5;

        const r_in = data[i];
        const g_in = data[i + 1];
        const b_in = data[i + 2];
        
        let r_processed = r_in;
        let g_processed = g_in;
        let b_processed = b_in;

        if (needsHueMasking) {
            const [h, s] = rgbToHsl(r_in, g_in, b_in);
            if (s > 0.05) {
                const r_adj = lutR[Math.round(Math.max(0, Math.min(255, r_in + dither)))];
                const g_adj = lutG[Math.round(Math.max(0, Math.min(255, g_in + dither)))];
                const b_adj = lutB[Math.round(Math.max(0, Math.min(255, b_in + dither)))];
                
                const distR = Math.min(Math.abs(h - 0), 360 - Math.abs(h - 0));
                const redInfluence = !isRedDefault && distR < 90 ? Math.pow((Math.cos((distR / 90) * Math.PI) + 1) / 2, 1.5) : 0;
                const distG = Math.min(Math.abs(h - 120), 360 - Math.abs(h - 120));
                const greenInfluence = !isGreenDefault && distG < 90 ? Math.pow((Math.cos((distG / 90) * Math.PI) + 1) / 2, 1.5) : 0;
                const distB = Math.min(Math.abs(h - 240), 360 - Math.abs(h - 240));
                const blueInfluence = !isBlueDefault && distB < 90 ? Math.pow((Math.cos((distB / 90) * Math.PI) + 1) / 2, 1.5) : 0;
                
                r_processed = r_in * (1 - redInfluence) + r_adj * redInfluence;
                g_processed = g_in * (1 - greenInfluence) + g_adj * greenInfluence;
                b_processed = b_in * (1 - blueInfluence) + b_adj * blueInfluence;
            }
        } else {
             r_processed = lutR[Math.round(Math.max(0, Math.min(255, r_in + dither)))];
             g_processed = lutG[Math.round(Math.max(0, Math.min(255, g_in + dither)))];
             b_processed = lutB[Math.round(Math.max(0, Math.min(255, b_in + dither)))];
        }

        const r_final_idx = Math.round(Math.max(0, Math.min(255, r_processed + dither)));
        const g_final_idx = Math.round(Math.max(0, Math.min(255, g_processed + dither)));
        const b_final_idx = Math.round(Math.max(0, Math.min(255, b_processed + dither)));
        
        data[i]   = lutRgb[r_final_idx];
        data[i+1] = lutRgb[g_final_idx];
        data[i+2] = lutRgb[b_final_idx];
    }
    return imageData;
}


const initialFilters: BasicFilters = {
    brightness: 100, contrast: 100, saturation: 100, sepia: 0,
    exposure: 0, highlights: 0, shadows: 0, whites: 0, blacks: 0,
    temperature: 0, vibrance: 0, sharpen: 0, dehaze: 0, grain: 0, haze: 0, hazeSpread: 50
};

const initialHSL: HSLFilters = Object.fromEntries(
    ['red', 'orange', 'yellow', 'green', 'aqua', 'blue', 'purple', 'magenta'].map(c => [c, { h: 0, s: 0, l: 0 }])
) as HSLFilters;

const initialCurve: Curve = [{ x: 0, y: 0 }, { x: 255, y: 255 }];
const initialCurves: CurvesState = {
    rgb: [...initialCurve],
    red: [...initialCurve],
    green: [...initialCurve],
    blue: [...initialCurve],
};

const initialAdjustments: Adjustments = {
    filters: { ...initialFilters },
    hsl: { ...initialHSL },
    curves: JSON.parse(JSON.stringify(initialCurves)),
};

const initialState: EditorState = {
    baseImageSrc: null,
    layers: [],
    adjustments: initialAdjustments,
};

const getInitialTheme = (): Theme => {
    const savedTheme = localStorage.getItem('photoEditorTheme');
    if (savedTheme && ['light', 'dark', 'mono-light', 'mono-dark'].includes(savedTheme)) {
        return savedTheme as Theme;
    }
    if (window.matchMedia?.('(prefers-color-scheme: dark)').matches) {
        return 'dark';
    }
    return 'light';
};

const MorphingLoader = () => (
    <svg className="morphing-loader" viewBox="0 0 100 100">
        <path d="M 50,10 A 40,40 0 1 1 50,90 A 40,40 0 1 1 50,10 Z" />
    </svg>
);

// Slider for live updates (fast filters)
const WavySlider = ({ label, unit, value, min, max, onChange, onDragEnd, disabled }: {label: string, unit: string, value: number, min: number, max: number, onChange: (value: number) => void, onDragEnd: () => void, disabled: boolean}) => {
    const [isDragging, setIsDragging] = useState(false);

    const handleStart = () => {
        if (disabled) return;
        setIsDragging(true);
    };

    const handleEnd = () => {
        if (isDragging) {
            setIsDragging(false);
            if (onDragEnd) onDragEnd();
        }
    };
    
    useEffect(() => {
        const endHandler = () => handleEnd();
        window.addEventListener('mouseup', endHandler);
        window.addEventListener('touchend', endHandler);
        return () => {
            window.removeEventListener('mouseup', endHandler);
            window.removeEventListener('touchend', endHandler);
        };
    }, [isDragging, onDragEnd]);

    const percentage = ((value - min) / (max - min)) * 100;
    const containerClasses = `wavy-slider-container ${isDragging ? 'dragging' : ''} ${disabled ? 'disabled' : ''}`;

    return (
        <div className={containerClasses}>
            <label>
                <span>{label}</span>
                <span>{value}{unit}</span>
            </label>
            <div className="wavy-slider-track-wrapper">
                 <div className="wavy-slider-fill" style={{ width: `${percentage}%` }}></div>
                 <input
                    type="range" min={min} max={max} value={value}
                    onChange={(e) => onChange(parseInt(e.target.value))}
                    onMouseDown={handleStart}
                    onTouchStart={handleStart}
                    disabled={disabled} aria-label={label}
                />
            </div>
        </div>
    );
};

// Slider for expensive filters that only updates on release
const CommitSlider = ({ label, unit, value: initialValue, min, max, onCommit, disabled }: {label: string, unit: string, value: number, min: number, max: number, onCommit: (value: number) => void, disabled: boolean}) => {
    const [value, setValue] = useState(initialValue);
    const [isDragging, setIsDragging] = useState(false);

    useEffect(() => {
        if (!isDragging) {
            setValue(initialValue);
        }
    }, [initialValue, isDragging]);

    const handleStart = () => {
        if (disabled) return;
        setIsDragging(true);
    };

    const handleEnd = useCallback(() => {
        if (isDragging) {
            setIsDragging(false);
            if (onCommit) {
                onCommit(value);
            }
        }
    }, [isDragging, onCommit, value]);
    
    useEffect(() => {
        const endHandler = () => handleEnd();
        window.addEventListener('mouseup', endHandler);
        window.addEventListener('touchend', endHandler);
        return () => {
            window.removeEventListener('mouseup', endHandler);
            window.removeEventListener('touchend', endHandler);
        };
    }, [handleEnd]);

    const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        setValue(parseInt(e.target.value));
    };

    const percentage = ((value - min) / (max - min)) * 100;
    const containerClasses = `wavy-slider-container ${isDragging ? 'dragging' : ''} ${disabled ? 'disabled' : ''}`;

    return (
        <div className={containerClasses}>
            <label>
                <span>{label}</span>
                <span>{value}{unit}</span>
            </label>
            <div className="wavy-slider-track-wrapper">
                 <div className="wavy-slider-fill" style={{ width: `${percentage}%` }}></div>
                 <input
                    type="range" min={min} max={max} value={value}
                    onChange={handleChange}
                    onMouseDown={handleStart}
                    onTouchStart={handleStart}
                    disabled={disabled} aria-label={label}
                />
            </div>
        </div>
    );
};
  
const CurvesEditor = ({ curves, onCurveChange, onCommit, disabled, histogram }: { curves: CurvesState, onCurveChange: (channel: keyof CurvesState, points: Curve) => void, onCommit: () => void, disabled: boolean, histogram: HistogramData | null }) => {
    const [activeChannel, setActiveChannel] = useState<keyof CurvesState>('rgb');
    const svgRef = useRef(null);
    const [draggingPoint, setDraggingPoint] = useState(null);
    const size = 255;

    const handleMouseDown = (e, index) => {
        e.stopPropagation();
        if (disabled) return;
        setDraggingPoint({ channel: activeChannel, index });
    };

    const handleMouseUp = () => {
        if (draggingPoint) {
            setDraggingPoint(null);
            onCommit();
        }
    };

    const handleMouseMove = (e) => {
        if (!draggingPoint || !svgRef.current) return;
        const rect = svgRef.current.getBoundingClientRect();
        let x = ((e.clientX - rect.left) / rect.width) * size;
        let y = ((e.clientY - rect.top) / rect.height) * size;
        x = Math.max(0, Math.min(size, x));
        y = Math.max(0, Math.min(size, y));

        if (draggingPoint.index === 0) x = 0;
        if (draggingPoint.index === curves[activeChannel].length - 1) x = size;
        
        let newPoints = [...curves[activeChannel]];
        newPoints[draggingPoint.index] = { x, y: size - y };
        newPoints.sort((a,b) => a.x - b.x);
        const newIndex = newPoints.findIndex(p => p.x === x && p.y === size - y);
        setDraggingPoint({ channel: activeChannel, index: newIndex >= 0 ? newIndex : draggingPoint.index });
        onCurveChange(activeChannel, newPoints);
    };

    const addPoint = (e) => {
        if (e.target.tagName === 'circle' || !svgRef.current || disabled) return;
        const rect = svgRef.current.getBoundingClientRect();
        const x = ((e.clientX - rect.left) / rect.width) * size;
        const y = ((e.clientY - rect.top) / rect.height) * size;
        const newPoints = [...curves[activeChannel], { x, y: size - y }];
        newPoints.sort((a,b) => a.x - b.x);
        onCurveChange(activeChannel, newPoints);
        onCommit();
    };

    const removePoint = (e, index) => {
        e.stopPropagation();
        if (index === 0 || index === curves[activeChannel].length - 1) return;
        const newPoints = curves[activeChannel].filter((_, i) => i !== index);
        onCurveChange(activeChannel, newPoints);
        onCommit();
    }
    
    const curvePathData = useMemo(() => {
        if (!curves || !curves[activeChannel]) return "M 0 255 L 255 0";
        const points = curves[activeChannel];
        if (points.length < 2) return "M 0 255 L 255 0";
        const lut = createLutFromPoints(points);
        return `M 0 ${size - lut[0]} ` + lut.map((y,x) => `L ${x} ${size - y}`).join(' ');
    }, [curves, activeChannel]);
    
    const histogramPathData = useMemo(() => {
        if (!histogram || !histogram[activeChannel]) return '';
        const data = histogram[activeChannel];
        return `M 0 ${size} ` + data.map((h, i) => `L ${i* (size / (data.length - 1))} ${size - h}`).join(' ') + ` L ${size} ${size} Z`;
    }, [histogram, activeChannel]);

    return (
        <div className="curves-editor">
            <div className="channel-selector">
                {(Object.keys(curves || {}) as Array<keyof CurvesState>).map(ch => (
                    <button key={ch} onClick={() => setActiveChannel(ch)} className={`${ch} ${activeChannel === ch ? 'active' : ''}`}>{ch.toUpperCase()}</button>
                ))}
            </div>
            <svg ref={svgRef} viewBox="0 0 255 255" className="curves-svg" onMouseMove={handleMouseMove} onMouseUp={handleMouseUp} onMouseLeave={handleMouseUp} onClick={addPoint}>
                <path d="M 64 0 L 64 255 M 128 0 L 128 255 M 192 0 L 192 255 M 0 64 L 255 64 M 0 128 L 255 128 M 0 192 L 255 192" className="grid-lines" />
                {histogramPathData && <path className="histogram-path" d={histogramPathData} />}
                <path d={curvePathData} className={`curve-path ${activeChannel}`} />
                {curves && curves[activeChannel] && curves[activeChannel].map((p, i) => (
                    <circle key={`${i}-${p.x}-${p.y}`} cx={p.x} cy={size - p.y} r="4" className="curve-point" onMouseDown={(e) => handleMouseDown(e, i)} onDoubleClick={(e) => removePoint(e, i)} />
                ))}
            </svg>
            <button className="btn btn-secondary" disabled={disabled} onClick={() => { onCurveChange(activeChannel, [...initialCurve]); onCommit(); }}>Reset Curve</button>
        </div>
    );
};


const App = () => {
  const [history, setHistory] = useState<HistoryState>({ stack: [initialState], index: 0 });
  
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [isProcessing, setIsProcessing] = useState<boolean>(false);
  const processingTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const [loadingMessage, setLoadingMessage] = useState<string>('');
  const [activeTab, setActiveTab] = useState<string | null>('basic');
  const [lastActiveAdjustTab, setLastActiveAdjustTab] = useState<string>('basic');
  const [activeLayerId, setActiveLayerId] = useState<string | null>(null);
  const [renamingLayerId, setRenamingLayerId] = useState<string | null>(null);
  
  const [compositionAdvice, setCompositionAdvice] = useState<string | null>(null);
  const [theme, setTheme] = useState<Theme>(getInitialTheme);
  const [showDownloadModal, setShowDownloadModal] = useState<boolean>(false);
  const [downloadOptions, setDownloadOptions] = useState<DownloadOptions>({ format: 'image/png', quality: 92 });
  const [showGenFillModal, setShowGenFillModal] = useState<boolean>(false);
  const [genFillPrompt, setGenFillPrompt] = useState<string>('');
  const [sketchPrompt, setSketchPrompt] = useState<string>('');
  const [customPresets, setCustomPresets] = useState<CustomPreset[]>([]);
  const [showMenu, setShowMenu] = useState(false);
  const [showThemes, setShowThemes] = useState(false);
  const [isPeeking, setIsPeeking] = useState(false);
  const [transform, setTransform] = useState<Transform>({ zoom: 1, pan: { x: 0, y: 0 } });
  const [showBetaMessage, setShowBetaMessage] = useState<boolean>(false);
  const [brushMode, setBrushMode] = useState<BrushMode>('none');
  const [brushSize, setBrushSize] = useState(30);
  const [histogram, setHistogram] = useState<HistogramData | null>(null);
  const [gradientPoints, setGradientPoints] = useState<{ start: Point; end: Point } | null>(null);
  const [isFullscreen, setIsFullscreen] = useState<boolean>(false);
  const [isMobilePanelCollapsed, setIsMobilePanelCollapsed] = useState<boolean>(false);


  const [isCropping, setIsCropping] = useState(false);
  const [isDraggingCrop, setIsDraggingCrop] = useState(false);
  const [cropBox, setCropBox] = useState<CropBox | null>(null);
  const [straightenAngle, setStraightenAngle] = useState(0);
  const [compositionOverlay, setCompositionOverlay] = useState<'none' | 'thirds' | 'spiral'>('none');
  const [aspectRatio, setAspectRatio] = useState('Original');
  const [customAspectRatioString, setCustomAspectRatioString] = useState('');
  
  const [liveAdjustments, setLiveAdjustments] = useState<Adjustments | null>(null);
  
  const currentState = history.stack[history.index];
  const activeAdjustments = liveAdjustments ?? currentState.adjustments;
  const activeLayer = useMemo(() => {
    return currentState.layers.find(l => l.id === activeLayerId);
  }, [activeLayerId, currentState.layers]);

  const canUndo = history.index > 0;
  const canRedo = history.index < history.stack.length - 1;
  const originalImageSrc = currentState.baseImageSrc;
  
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const maskCanvasRef = useRef<HTMLCanvasElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const canvasWrapperRef = useRef<HTMLDivElement>(null);
  const appWrapperRef = useRef<HTMLDivElement>(null);
  const panState = useRef({ isPanning: false, startX: 0, startY: 0 });
  const drawingState = useRef({ isDrawing: false, path: [] as Point[] });
  const cropDragInfo = useRef<{ handle: string; startX: number; startY: number; initialRect: CropBox } | null>(null);

  const subNavRef = useRef<HTMLDivElement>(null);
  const subNavButtonRefs = useRef<Map<string, HTMLButtonElement | null>>(new Map());

  const ai = new GoogleGenAI({ apiKey: API_KEY });

  const KiteIcon = () => (
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" style={{width: '1.75rem', height: '1.75rem'}}>
      <path d="M12 2C19 5, 22 7, 22 12C22 17, 19 19, 12 22C5 19, 2 17, 2 12C2 7, 5 5, 12 2Z" />
    </svg>
  );

  const TABS = [
    { id: 'presets', title: 'Presets', icon: 'auto_awesome' },
    { id: 'basic', title: 'Basic Adjustments', icon: 'tune' },
    { id: 'curves', title: 'Curves', icon: 'show_chart' },
    { id: 'transform', title: 'Crop & Rotate', icon: 'crop_rotate' },
    { id: 'masking', title: 'Masking & Layers', icon: 'layers' },
    { id: 'color', title: 'Color', icon: 'palette' },
    { id: 'effects', title: 'Effects', icon: 'movie_filter' },
    { id: 'ai-tools', title: 'AI Tools', icon: <KiteIcon /> },
  ];

  const getPointOnImage = useCallback((clientX: number, clientY: number): Point | null => {
      if (!canvasWrapperRef.current || !canvasRef.current) return null;
      const wrapperRect = canvasWrapperRef.current.getBoundingClientRect();
      const canvas = canvasRef.current;
      
      const mouseX = clientX - wrapperRect.left;
      const mouseY = clientY - wrapperRect.top;
      
      let imageX = (mouseX - transform.pan.x) / transform.zoom;
      let imageY = (mouseY - transform.pan.y) / transform.zoom;

      const currentAngle = isCropping ? 0 : straightenAngle;
      if (currentAngle !== 0) {
        const rad = -currentAngle * Math.PI / 180;
        const cos = Math.cos(rad);
        const sin = Math.sin(rad);
        const centerX = canvas.width / 2;
        const centerY = canvas.height / 2;
        
        const translatedX = imageX - centerX;
        const translatedY = imageY - centerY;

        const rotatedX = translatedX * cos - translatedY * sin;
        const rotatedY = translatedX * sin + translatedY * cos;

        imageX = rotatedX + centerX;
        imageY = rotatedY + centerY;
      }
      
      return { x: imageX, y: imageY };
  }, [transform, straightenAngle, isCropping]);
  
  const commitChange = (updater: (prevState: EditorState) => EditorState, overwrite: boolean = false) => {
    setHistory(prevHistoryState => {
        const { stack, index } = prevHistoryState;
        
        const baseState = stack[index];
        const newState = updater(baseState);

        const newHistoryBase = overwrite ? stack.slice(0, index) : stack.slice(0, index + 1);
        
        let newStack = [...newHistoryBase, newState];

        if (newStack.length > MAX_HISTORY) {
            newStack.shift();
        }

        return {
            stack: newStack,
            index: newStack.length - 1,
        };
    });
  };
  
  const undo = () => {
    if (canUndo) {
        setLiveAdjustments(null);
        setHistory(h => ({ ...h, index: h.index - 1 }));
    }
  };
  const redo = () => {
    if (canRedo) {
        setLiveAdjustments(null);
        setHistory(h => ({ ...h, index: Math.min(h.stack.length - 1, h.index + 1) }));
    }
  };

  useEffect(() => {
    const savedPresets = localStorage.getItem('photoEditorPresets');
    if (savedPresets) setCustomPresets(JSON.parse(savedPresets));
    
    const hasSeenBetaMessage = localStorage.getItem('hasSeenBetaMessage');
    if (!hasSeenBetaMessage) setShowBetaMessage(true);
  }, []);

  useEffect(() => { document.body.className = `${theme}-theme`; }, [theme]);
  
  useEffect(() => {
      const mediaQuery = window.matchMedia('(prefers-color-scheme: dark)');
      const handleChange = (e: MediaQueryListEvent | { matches: boolean }) => {
          if (localStorage.getItem('photoEditorTheme') === null) {
              setTheme(e.matches ? 'dark' : 'light');
          }
      };

      if (mediaQuery.addEventListener) {
          mediaQuery.addEventListener('change', handleChange as (e: Event) => void);
          return () => mediaQuery.removeEventListener('change', handleChange as (e: Event) => void);
      } else {
          // Deprecated fallback for older browsers
          mediaQuery.addListener(handleChange as any);
          return () => mediaQuery.removeListener(handleChange as any);
      }
  }, []);

  const [indicatorStyle, setIndicatorStyle] = useState({});
  useEffect(() => {
    const currentTabId = activeTab === 'presets' ? lastActiveAdjustTab : activeTab;
    const currentBtn = subNavButtonRefs.current.get(currentTabId);
    const nav = subNavRef.current;

    if (currentBtn && nav) {
        currentBtn.scrollIntoView({
            behavior: 'smooth',
            inline: 'center',
            block: 'nearest'
        });

        requestAnimationFrame(() => {
            const navRect = nav.getBoundingClientRect();
            const btnRect = currentBtn.getBoundingClientRect();
            
            const left = btnRect.left - navRect.left + nav.scrollLeft + btnRect.width / 2;
            const top = btnRect.top - navRect.top + btnRect.height / 2;
            
            setIndicatorStyle({ left: `${left}px`, top: `${top}px` });
        });
    }
  }, [activeTab, lastActiveAdjustTab, originalImageSrc]);
  
  useEffect(() => {
    const nav = subNavRef.current;
    if (!nav) return;

    const handleWheel = (e: WheelEvent) => {
        if (e.deltaY === 0) return;
        e.preventDefault();
        nav.scrollBy({
            left: e.deltaY,
            behavior: 'auto' // Use auto for better responsiveness with mouse wheels
        });
    };

    nav.addEventListener('wheel', handleWheel, { passive: false });
    return () => nav.removeEventListener('wheel', handleWheel);
  }, []);
  
  const toggleFullscreen = () => {
      const elem = appWrapperRef.current;
      if (!elem) return;

      if (!document.fullscreenElement) {
          elem.requestFullscreen().catch(err => {
              alert(`Error attempting to enable full-screen mode: ${err.message} (${err.name})`);
          });
      } else {
          if (document.exitFullscreen) {
              document.exitFullscreen();
          }
      }
  };

  useEffect(() => {
      const handleFullscreenChange = () => {
          setIsFullscreen(!!document.fullscreenElement);
      };
      document.addEventListener('fullscreenchange', handleFullscreenChange);
      return () => document.removeEventListener('fullscreenchange', handleFullscreenChange);
  }, []);


  const handleThemeChange = (newTheme: Theme) => {
      setTheme(newTheme);
      localStorage.setItem('photoEditorTheme', newTheme);
  };
  
  function curvesAreDefault(curves: CurvesState) {
    if (!curves) return true;
    const isDefault = (c: Curve) => c.length === 2 && c[0].x === 0 && c[0].y === 0 && c[1].x === 255 && c[1].y === 255;
    return isDefault(curves.rgb) && isDefault(curves.red) && isDefault(curves.green) && isDefault(curves.blue);
  }
  
  const getAdjustmentsCss = (adjustments: Adjustments) => {
      const { filters, hsl, curves } = adjustments;
      const { brightness, contrast, saturation, sepia, ...pixelFilters } = filters;
      const filterString = `brightness(${brightness}%) contrast(${contrast}%) saturate(${saturation}%) sepia(${sepia}%)`;
      
      const needsPixelManipulation = Object.values(pixelFilters).some(v => v !== initialFilters[Object.keys(pixelFilters).find(k => pixelFilters[k] === v) as keyof typeof pixelFilters])
          || !curvesAreDefault(curves)
          || Object.values(hsl).some(c => c.h !== 0 || c.s !== 0 || c.l !== 0);

      return { needsPixelManipulation, filterString, pixelFilters: { ...pixelFilters, hsl, curves } };
  }
  
  const loadImage = (src: string): Promise<HTMLImageElement> => new Promise((resolve, reject) => {
      const img = new Image();
      img.crossOrigin = 'anonymous';
      img.onload = () => resolve(img);
      img.onerror = reject;
      img.src = src;
  });

  const drawCanvas = useCallback(async () => {
    const canvas = canvasRef.current;
    const ctx = canvas?.getContext('2d', { willReadFrequently: true });
    if (!canvas || !ctx || !currentState.baseImageSrc) return;

    try {
        const baseImage = await loadImage(currentState.baseImageSrc);
        if (canvas.width !== baseImage.naturalWidth || canvas.height !== baseImage.naturalHeight) {
            canvas.width = baseImage.naturalWidth;
            canvas.height = baseImage.naturalHeight;
        }

        // Create a separate canvas for the fully adjusted image
        const adjustedImageCanvas = document.createElement('canvas');
        adjustedImageCanvas.width = canvas.width;
        adjustedImageCanvas.height = canvas.height;
        const adjustedCtx = adjustedImageCanvas.getContext('2d', { willReadFrequently: true });
        if (!adjustedCtx) throw new Error("Could not create adjusted image context");

        const { needsPixelManipulation, filterString, pixelFilters } = getAdjustmentsCss(activeAdjustments);

        // Apply fast CSS filters
        adjustedCtx.filter = filterString;
        adjustedCtx.drawImage(baseImage, 0, 0);
        adjustedCtx.filter = 'none';

        if (needsPixelManipulation) {
            let imageData = adjustedCtx.getImageData(0, 0, canvas.width, canvas.height);
            const { exposure, temperature, vibrance, dehaze, highlights, shadows, whites, blacks, sharpen, grain, haze, hazeSpread, hsl, curves } = pixelFilters;
            if (exposure !== 0) imageData = applyExposure(imageData, exposure);
            if (temperature !== 0) imageData = applyTemperature(imageData, temperature);
            if (vibrance !== 0) imageData = applyVibrance(imageData, vibrance);
            if (dehaze !== 0) imageData = applyDehaze(imageData, dehaze);
            if (highlights !== 0 || shadows !== 0) imageData = applyHighlightsShadows(imageData, highlights, shadows);
            if (whites !== 0 || blacks !== 0) imageData = applyWhitesBlacks(imageData, whites, blacks);
            if (!curvesAreDefault(curves)) imageData = applyCurves(imageData, curves);
            if (Object.values(hsl).some(c => c.h !== 0 || c.s !== 0 || c.l !== 0)) imageData = applyHsl(imageData, hsl);
            if (haze > 0) imageData = applyHaze(imageData, haze, hazeSpread);
            if (sharpen > 0) imageData = applySharpen(imageData, sharpen);
            if (grain > 0) imageData = applyGrain(imageData, grain);
            adjustedCtx.putImageData(imageData, 0, 0);
        }

        // --- Compositing Stage ---
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        const visibleLayers = currentState.layers.filter(l => l.isVisible && l.maskSrc);

        if (visibleLayers.length === 0) {
            // NO MASK: Draw the globally adjusted image directly.
            ctx.drawImage(adjustedImageCanvas, 0, 0);
        } else {
            // MASK(S) EXIST: Composite the adjusted image over the base image using the combined mask.
            
            // 1. Draw original image as the base layer.
            ctx.drawImage(baseImage, 0, 0);

            // 2. Create a single composite mask from all visible layers.
            const maskCompositeCanvas = document.createElement('canvas');
            maskCompositeCanvas.width = canvas.width;
            maskCompositeCanvas.height = canvas.height;
            const maskCtx = maskCompositeCanvas.getContext('2d');
            if (!maskCtx) throw new Error("Could not create mask context");

            maskCtx.fillStyle = 'black';
            maskCtx.fillRect(0, 0, canvas.width, canvas.height);

            // Combine all visible masks
            for (const layer of visibleLayers) {
                const maskImage = await loadImage(layer.maskSrc!);
                
                const tempMaskCanvas = document.createElement('canvas');
                tempMaskCanvas.width = canvas.width;
                tempMaskCanvas.height = canvas.height;
                const tempMaskCtx = tempMaskCanvas.getContext('2d')!;
                
                tempMaskCtx.drawImage(maskImage, 0, 0);
                
                if (layer.isMaskInverted) {
                    tempMaskCtx.globalCompositeOperation = 'xor';
                    tempMaskCtx.fillStyle = 'white';
                    tempMaskCtx.fillRect(0, 0, canvas.width, canvas.height);
                }
                
                maskCtx.globalCompositeOperation = 'lighten';
                maskCtx.drawImage(tempMaskCanvas, 0, 0);
            }
            
            // 3. Apply the composite mask to the adjusted image and draw it on top.
            const finalAdjustedLayerCanvas = document.createElement('canvas');
            finalAdjustedLayerCanvas.width = canvas.width;
            finalAdjustedLayerCanvas.height = canvas.height;
            const finalAdjustedLayerCtx = finalAdjustedLayerCanvas.getContext('2d')!;
            
            finalAdjustedLayerCtx.drawImage(adjustedImageCanvas, 0, 0);
            finalAdjustedLayerCtx.globalCompositeOperation = 'destination-in';
            finalAdjustedLayerCtx.drawImage(maskCompositeCanvas, 0, 0);
            
            // 4. Draw the final masked layer over the base image on the main canvas.
            ctx.globalCompositeOperation = 'source-over';
            ctx.drawImage(finalAdjustedLayerCanvas, 0, 0);
        }

    } catch (error) {
        console.error("Error drawing canvas:", error);
    }
  }, [currentState, activeAdjustments]);

  const drawCanvasRef = useRef(drawCanvas);
  useEffect(() => { drawCanvasRef.current = drawCanvas; });

  const startProcessing = () => {
    if (processingTimerRef.current) clearTimeout(processingTimerRef.current);
    processingTimerRef.current = setTimeout(() => {
        setIsProcessing(true);
    }, 500);
  };
  const endProcessing = () => {
    if (processingTimerRef.current) clearTimeout(processingTimerRef.current);
    setIsProcessing(false);
  };

  useEffect(() => {
    let isCancelled = false;
    const redraw = async () => {
      await drawCanvasRef.current();
      if (!isCancelled) {
        endProcessing();
      }
    };
    const handle = requestAnimationFrame(redraw);
    return () => {
        isCancelled = true;
        cancelAnimationFrame(handle);
    };
  }, [currentState, activeAdjustments]);
  
  const calculateHistogram = useCallback((imageData: ImageData): HistogramData => {
        const rgb = new Array(256).fill(0);
        const r = new Array(256).fill(0);
        const g = new Array(256).fill(0);
        const b = new Array(256).fill(0);
        const data = imageData.data;

        for (let i = 0; i < data.length; i += 4) {
            const red = data[i];
            const green = data[i+1];
            const blue = data[i+2];
            const luma = Math.round(red * 0.299 + green * 0.587 + blue * 0.114);

            rgb[luma]++;
            r[red]++;
            g[green]++;
            b[blue]++;
        }
        
        const normalize = (hist: number[]) => {
            const maxVal = Math.max(...hist);
            if (maxVal === 0) return hist;
            const scale = 255 / maxVal;
            return hist.map(val => val * scale);
        };

        return { rgb: normalize(rgb), red: normalize(r), green: normalize(g), blue: normalize(b) };
  }, []);

  useEffect(() => {
    if (activeTab !== 'curves' || !canvasRef.current || !canvasRef.current.width || isLoading) {
      if (histogram !== null) setHistogram(null);
      return;
    }
    const ctx = canvasRef.current.getContext('2d', { willReadFrequently: true });
    if (!ctx) return;
    
    const imageData = ctx.getImageData(0, 0, canvasRef.current.width, canvasRef.current.height);
    const histData = calculateHistogram(imageData);
    setHistogram(histData);

  }, [activeTab, currentState, activeAdjustments, isLoading, calculateHistogram]);
  
    useEffect(() => {
        const maskCanvas = maskCanvasRef.current;
        const ctx = maskCanvas?.getContext('2d');
        if (!ctx || !maskCanvas) return;
        
        ctx.clearRect(0, 0, maskCanvas.width, maskCanvas.height);

        if (drawingState.current.isDrawing && gradientPoints && (brushMode === 'linear-gradient' || brushMode === 'radial-gradient')) {
            if (brushMode === 'linear-gradient') {
                const grad = ctx.createLinearGradient(gradientPoints.start.x, gradientPoints.start.y, gradientPoints.end.x, gradientPoints.end.y);
                grad.addColorStop(0, 'white');
                grad.addColorStop(1, 'transparent');
                ctx.fillStyle = grad;
                ctx.fillRect(0, 0, maskCanvas.width, maskCanvas.height);
            } else if (brushMode === 'radial-gradient') {
                const dx = gradientPoints.end.x - gradientPoints.start.x;
                const dy = gradientPoints.end.y - gradientPoints.start.y;
                const radius = Math.sqrt(dx * dx + dy * dy);
                if (radius > 0) {
                    const grad = ctx.createRadialGradient(gradientPoints.start.x, gradientPoints.start.y, 0, gradientPoints.start.x, gradientPoints.start.y, radius);
                    grad.addColorStop(0, 'white');
                    grad.addColorStop(1, 'transparent');
                    ctx.fillStyle = grad;
                    ctx.fillRect(0, 0, maskCanvas.width, maskCanvas.height);
                }
            }
        } 
        else if (brushMode !== 'none' && activeLayer?.maskSrc) {
            const maskImg = new Image();
            maskImg.src = activeLayer.maskSrc;
            maskImg.onload = () => { ctx.drawImage(maskImg, 0, 0, maskCanvas.width, maskCanvas.height); }
        }
    }, [brushMode, activeLayer, gradientPoints]);

  const handleLiveUpdate = (updater: (draft: Adjustments) => Adjustments) => {
    setLiveAdjustments(updater(activeAdjustments));
  };
  
  const handleCommitUpdate = () => {
    if (liveAdjustments) {
        startProcessing();
        setTimeout(() => {
            commitChange(draft => ({
                ...draft,
                adjustments: liveAdjustments
            }));
            setLiveAdjustments(null);
        }, 10);
    }
  };

  const handleFilterChange = (filter: keyof BasicFilters, value: number) => {
    handleLiveUpdate(draft => ({ ...draft, filters: { ...draft.filters, [filter]: value } }));
  };
  
  const handleHslChange = (color: keyof HSLFilters, type: 'h' | 's' | 'l', value: number) => {
    handleLiveUpdate(draft => ({ ...draft, hsl: { ...draft.hsl, [color]: { ...draft.hsl[color], [type]: value } } }));
  };
  
  const handleFilterCommit = (filter: keyof BasicFilters, value: number) => {
    startProcessing();
    setTimeout(() => {
        commitChange(draft => ({
            ...draft,
            adjustments: { ...draft.adjustments, filters: { ...draft.adjustments.filters, [filter]: value } }
        }));
    }, 10);
    setLiveAdjustments(null);
  };
  
  const handleHslCommit = (color: keyof HSLFilters, type: 'h' | 's' | 'l', value: number) => {
    startProcessing();
    setTimeout(() => {
         commitChange(draft => ({
            ...draft,
            adjustments: { ...draft.adjustments, hsl: { ...draft.adjustments.hsl, [color]: { ...draft.adjustments.hsl[color], [type]: value } } }
        }));
    }, 10);
    setLiveAdjustments(null);
  };

  const handleCurveChange = (channel: keyof CurvesState, points: Curve) => {
    handleLiveUpdate(draft => ({ ...draft, curves: { ...draft.curves, [channel]: points } }));
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const reader = new FileReader();
      reader.onload = (event) => {
        const imgSrc = event.target?.result as string;
        const newInitialState: EditorState = { ...initialState, baseImageSrc: imgSrc };
        setHistory({ stack: [newInitialState], index: 0 });
        setActiveLayerId(null);
        setLiveAdjustments(null);
        setTransform({ zoom: 1, pan: { x: 0, y: 0 } });
        setActiveTab('basic');
        setCompositionAdvice(null);
        const img = new Image();
        img.src = imgSrc;
        img.onload = () => {
          if (maskCanvasRef.current) {
            maskCanvasRef.current.width = img.naturalWidth;
            maskCanvasRef.current.height = img.naturalHeight;
          }
          fitToScreen();
        }
      };
      reader.readAsDataURL(e.target.files[0]);
    }
  };

  const exportImage = async () => {
    if (!canvasRef.current) return;
    await drawCanvasRef.current(); // Ensure canvas is up-to-date
    const dataUrl = canvasRef.current.toDataURL(downloadOptions.format, downloadOptions.format === 'image/jpeg' ? downloadOptions.quality / 100 : undefined);
    const link = document.createElement('a');
    link.href = dataUrl;
    link.download = `edited-image.${downloadOptions.format.split('/')[1]}`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    setShowDownloadModal(false);
  };
  
  const resetAll = () => {
    if (!originalImageSrc) return;
    const resetState: EditorState = {
        ...initialState,
        baseImageSrc: originalImageSrc,
    };
    setHistory({ stack: [resetState], index: 0 });
    setLiveAdjustments(null);
    setActiveLayerId(null);
    setActiveTab('basic');
  };

  const savePreset = () => {
    const name = prompt("Enter preset name:");
    if (name) {
      const newPreset: CustomPreset = {
        name,
        adjustments: currentState.adjustments
      };
      const updatedPresets = [...customPresets, newPreset];
      setCustomPresets(updatedPresets);
      localStorage.setItem('photoEditorPresets', JSON.stringify(updatedPresets));
    }
  };

  const applyPreset = (preset: CustomPreset) => {
    commitChange(draft => ({ ...draft, adjustments: preset.adjustments }));
  };
  
  const deletePreset = (index: number) => {
    const updatedPresets = customPresets.filter((_, i) => i !== index);
    setCustomPresets(updatedPresets);
    localStorage.setItem('photoEditorPresets', JSON.stringify(updatedPresets));
  }

  const applyAiAction = async (action: string, prompt?: string) => {
      if (!currentState.baseImageSrc || isLoading) return;
      setIsLoading(true);
      setLoadingMessage(`AI is ${action.replace(/-/g, ' ')}...`);
      setCompositionAdvice(null);
      const maskDataUrl = activeLayer?.maskSrc;

      try {
          await drawCanvasRef.current(); // Flatten image for context
          const imageBase64 = canvasRef.current.toDataURL().split(',')[1];
          const maskBase64 = maskDataUrl ? maskDataUrl.split(',')[1] : undefined;

          let contents: any = {
              parts: [
                  { inlineData: { mimeType: 'image/png', data: imageBase64 } },
                  { text: `Action: ${action}.` }
              ]
          };

          if (prompt) contents.parts.push({ text: `Prompt: ${prompt}` });
          if (maskBase64) {
              contents.parts.push({ inlineData: { mimeType: 'image/png', data: maskBase64 } });
              contents.parts.push({ text: "The second image is a mask..." });
          }

          const response = await ai.models.generateContent({
              model: 'gemini-2.5-flash',
              contents,
              config: { responseMimeType: "application/json", responseSchema: {
                  type: 'OBJECT', properties: {
                      image: { type: 'STRING' },
                      analysis: { type: 'STRING' },
                  }
              }}
          });
          
          const result = JSON.parse(response.text.trim());
          if (result.image) {
              const newImageSrc = `data:image/png;base64,${result.image}`;
              commitChange(() => ({ ...initialState, baseImageSrc: newImageSrc }));
              setActiveLayerId(null);
          }
          if (result.analysis) setCompositionAdvice(result.analysis);

      } catch (error) {
          console.error("AI Action Failed:", error);
          setCompositionAdvice("Sorry, the AI couldn't process that.");
      } finally {
          setIsLoading(false);
          setBrushMode('none');
          setShowGenFillModal(false);
          setGenFillPrompt(''); setSketchPrompt('');
      }
  };
  
    const handleGenerateAiMask = async (type: 'subject' | 'sky') => {
        if (!currentState.baseImageSrc || isLoading) return;
        
        const layerName = type === 'subject' ? 'Subject Mask' : 'Sky Mask';
        const newLayerId = addLayer(layerName);

        setIsLoading(true);
        setLoadingMessage(`AI is selecting the ${type}...`);

        try {
            await drawCanvasRef.current(); 
            const imageBase64 = canvasRef.current.toDataURL('image/png').split(',')[1];
            
            const prompt = `You are a sophisticated photo editing AI. Your task is to perform segmentation. Analyze the provided image and generate a precise, black and white segmentation mask of the main ${type}. The area of the ${type} should be pure white (#FFFFFF) and everything else should be pure black (#000000). Output the mask as a base64 encoded PNG image. The output must only be the JSON object.`;
            
            const response = await ai.models.generateContent({
                model: 'gemini-2.5-flash',
                contents: { parts: [
                    { inlineData: { mimeType: 'image/png', data: imageBase64 } },
                    { text: prompt }
                ]},
                config: {
                    responseMimeType: "application/json",
                    responseSchema: {
                        type: 'OBJECT',
                        properties: {
                            mask: { type: 'STRING', description: 'The base64 encoded PNG mask image.' }
                        }
                    }
                }
            });

            const result = JSON.parse(response.text.trim());

            if (result.mask) {
                const newMaskSrc = `data:image/png;base64,${result.mask}`;
                commitChange(draft => ({
                    ...draft,
                    layers: draft.layers.map(l => l.id === newLayerId ? { ...l, maskSrc: newMaskSrc } : l)
                }));
            } else {
                throw new Error("AI did not return a mask.");
            }

        } catch (error) {
            console.error("AI Mask Generation Failed:", error);
            deleteLayer(newLayerId);
            alert(`Sorry, the AI couldn't select the ${type}. Please try again.`);
        } finally {
            setIsLoading(false);
            setLoadingMessage('');
        }
    };

  const handleTabClick = (tabId: string) => {
      if (isCropping) handleExitCropMode();

      if (tabId === 'presets') {
          setActiveTab(activeTab === 'presets' ? lastActiveAdjustTab : 'presets');
      } else {
          setActiveTab(tabId);
          if (tabId !== 'masking' && tabId !== 'transform') {
              setLastActiveAdjustTab(tabId);
          }
      }
  };
  
  const handleInteractionAnimation = (e: React.MouseEvent<HTMLElement>) => {
    const el = e.currentTarget;

    // Add bounce to the clicked element
    el.classList.add('nudge-main');
    setTimeout(() => el.classList.remove('nudge-main'), 500);

    const prev1 = el.previousElementSibling as HTMLElement;
    const next1 = el.nextElementSibling as HTMLElement;
    if (prev1) {
        prev1.classList.add('nudge-down-1');
        setTimeout(() => prev1.classList.remove('nudge-down-1'), 500);
        const prev2 = prev1.previousElementSibling as HTMLElement;
        if (prev2) {
            prev2.classList.add('nudge-down-2');
            setTimeout(() => prev2.classList.remove('nudge-down-2'), 500);
        }
    }
    if (next1) {
        next1.classList.add('nudge-down-1');
        setTimeout(() => next1.classList.remove('nudge-down-1'), 500);
        const next2 = next1.nextElementSibling as HTMLElement;
        if (next2) {
            next2.classList.add('nudge-down-2');
            setTimeout(() => next2.classList.remove('nudge-down-2'), 500);
        }
    }
  };

  const handlePanStart = (e: React.MouseEvent | React.TouchEvent) => {
      if (brushMode !== 'none' || isCropping) return;
      const clientX = 'touches' in e ? e.touches[0].clientX : e.clientX;
      const clientY = 'touches' in e ? e.touches[0].clientY : e.clientY;
      panState.current = { isPanning: true, startX: clientX - transform.pan.x, startY: clientY - transform.pan.y };
      if (canvasWrapperRef.current) canvasWrapperRef.current.style.cursor = 'grabbing';
  };
  
  const handlePanMove = (e: React.MouseEvent | React.TouchEvent) => {
      if (!panState.current.isPanning) return;
      const clientX = 'touches' in e ? e.touches[0].clientX : e.clientX;
      const clientY = 'touches' in e ? e.touches[0].clientY : e.clientY;
      setTransform(t => ({ ...t, pan: { x: clientX - panState.current.startX, y: clientY - panState.current.startY } }));
  };

  const handlePanEnd = () => {
      panState.current.isPanning = false;
      if (canvasWrapperRef.current) canvasWrapperRef.current.style.cursor = brushMode !== 'none' ? 'crosshair' : 'grab';
  };
  
  const handleZoom = (e: React.WheelEvent) => {
      if (!canvasWrapperRef.current) return;
      const rect = canvasWrapperRef.current.getBoundingClientRect();
      const mouseX = e.clientX - rect.left;
      const mouseY = e.clientY - rect.top;
      const newZoom = Math.max(0.1, Math.min(5, transform.zoom - e.deltaY * 0.001));
      
      const newPanX = mouseX - (mouseX - transform.pan.x) * (newZoom / transform.zoom);
      const newPanY = mouseY - (mouseY - transform.pan.y) * (newZoom / transform.zoom);
      setTransform({ zoom: newZoom, pan: { x: newPanX, y: newPanY } });
  };
  
  const zoomWithButtons = (direction: 'in' | 'out') => {
    const zoomFactor = 1.2;
    const oldZoom = transform.zoom;
    const newZoom = direction === 'in' ? oldZoom * zoomFactor : oldZoom / zoomFactor;
    handleSliderZoom(newZoom);
  };
  
  const handleSliderZoom = (newZoomValue: number) => {
    if (!canvasWrapperRef.current) return;
    const wrapper = canvasWrapperRef.current;
    const anchorX = wrapper.clientWidth / 2;
    const anchorY = wrapper.clientHeight / 2;
    const oldZoom = transform.zoom;
    const newZoom = Math.max(0.1, Math.min(5, newZoomValue));
    if (Math.abs(oldZoom - newZoom) < 0.001) return;
    const newPanX = anchorX - (anchorX - transform.pan.x) * (newZoom / oldZoom);
    const newPanY = anchorY - (anchorY - transform.pan.y) * (newZoom / oldZoom);
    setTransform({ zoom: newZoom, pan: { x: newPanX, y: newPanY } });
  };
  
  const fitToScreen = useCallback(() => {
    if (!canvasRef.current || !canvasWrapperRef.current || !canvasRef.current.width) return;
    const canvas = canvasRef.current;
    const wrapper = canvasWrapperRef.current;
    const zoomX = wrapper.clientWidth / canvas.width;
    const zoomY = wrapper.clientHeight / canvas.height;
    const newZoom = Math.min(zoomX, zoomY) * 0.95;
    
    setTransform({
        zoom: newZoom,
        pan: {
            x: (wrapper.clientWidth - canvas.width * newZoom) / 2,
            y: (wrapper.clientHeight - canvas.height * newZoom) / 2
        }
    });
  }, []);

  const handleCanvasMouseDown = (e: React.MouseEvent) => {
      if (isCropping) return;
      if (brushMode === 'none') { handlePanStart(e); return; }
      
      const isGradientTool = brushMode === 'linear-gradient' || brushMode === 'radial-gradient';
      
      if (!isGradientTool && !activeLayerId) { 
          alert("Please add and select a mask layer first."); return; 
      }

      drawingState.current.isDrawing = true;
      const point = getPointOnImage(e.clientX, e.clientY);
      if (!point) return;
      
      if (isGradientTool) {
          const layerName = brushMode === 'linear-gradient' ? 'Linear Gradient' : 'Radial Gradient';
          addLayer(layerName); // This will also set activeLayerId
          setGradientPoints({ start: point, end: point });
      } else {
          drawingState.current.path = [point];
      }
  };

  const handleCanvasMouseMove = (e: React.MouseEvent) => {
      if (isCropping) return;
      if (brushMode === 'none') { handlePanMove(e); return; }
      if (!drawingState.current.isDrawing) return;
      
      const point = getPointOnImage(e.clientX, e.clientY);
      if (!point) return;

      if (brushMode === 'linear-gradient' || brushMode === 'radial-gradient') {
          setGradientPoints(p => p ? { ...p, end: point } : null);
      } else {
          const newPath = [...drawingState.current.path, point];
          drawingState.current.path = newPath;
          drawBrushStroke(newPath);
      }
  };

  const handleCanvasMouseUp = () => {
      if (isCropping) return;
      if (brushMode === 'none') { handlePanEnd(); return; }
      if (!drawingState.current.isDrawing) return;

      drawingState.current.isDrawing = false;
      const maskCanvas = maskCanvasRef.current;
      if (maskCanvas && activeLayerId) {
          const maskDataUrl = maskCanvas.toDataURL();
          commitChange(draft => ({
              ...draft,
              layers: draft.layers.map(l => l.id === activeLayerId ? { ...l, maskSrc: maskDataUrl } : l)
          }));

          if (brushMode === 'magic-remove') {
              applyAiAction('magic-remove');
          }
      }
      
      if (brushMode === 'linear-gradient' || brushMode === 'radial-gradient') {
          setGradientPoints(null);
          setBrushMode('none');
      }
  };

  const drawBrushStroke = (path: Point[]) => {
      const maskCanvas = maskCanvasRef.current;
      const ctx = maskCanvas?.getContext('2d');
      if (!ctx || path.length < 2) return;

      ctx.globalCompositeOperation = brushMode === 'eraser' ? 'destination-out' : 'source-over';
      ctx.beginPath();
      ctx.moveTo(path[0].x, path[0].y);
      for (let i = 1; i < path.length; i++) {
          ctx.lineTo(path[i].x, path[i].y);
      }
      ctx.strokeStyle = 'white';
      ctx.lineWidth = brushSize;
      ctx.lineCap = 'round';
      ctx.lineJoin = 'round';
      ctx.stroke();
  };
  
  const clearActiveMask = () => {
    if (!activeLayerId) return;
    commitChange(draft => ({
      ...draft,
      layers: draft.layers.map(l => l.id === activeLayerId ? { ...l, maskSrc: null } : l)
    }));
  };

    const applyDestructiveTransform = async (transformFunction: (canvas: HTMLCanvasElement, ctx: CanvasRenderingContext2D, img: HTMLImageElement) => void) => {
        if (!currentState.baseImageSrc) return;
        setIsLoading(true);
        setLoadingMessage('Applying transform...');

        const flattenedCanvas = document.createElement('canvas');
        flattenedCanvas.width = canvasRef.current.width;
        flattenedCanvas.height = canvasRef.current.height;
        const flattenedCtx = flattenedCanvas.getContext('2d');
        flattenedCtx.drawImage(canvasRef.current, 0, 0);
        const flattenedSrc = flattenedCanvas.toDataURL();
        
        const img = new Image();
        img.crossOrigin = "anonymous";
        img.src = flattenedSrc;
        img.onload = () => {
            const tempCanvas = document.createElement('canvas');
            const tempCtx = tempCanvas.getContext('2d');
            if (!tempCtx) { setIsLoading(false); return; }
            
            transformFunction(tempCanvas, tempCtx, img);
            const newDataUrl = tempCanvas.toDataURL();
            
            commitChange((draft) => ({ ...draft, baseImageSrc: newDataUrl, layers: [], adjustments: initialAdjustments }), true);
            setActiveLayerId(null);

            const newImg = new Image();
            newImg.src = newDataUrl;
            newImg.onload = () => {
                if(maskCanvasRef.current) {
                    maskCanvasRef.current.width = newImg.width;
                    maskCanvasRef.current.height = newImg.height;
                }
                fitToScreen();
                setIsLoading(false);
            };
        };
        img.onerror = () => setIsLoading(false);
    };

    const handleRotate = (angle: 90 | -90) => {
        applyDestructiveTransform((canvas, ctx, img) => {
            canvas.width = img.height;
            canvas.height = img.width;
            ctx.translate(canvas.width / 2, canvas.height / 2);
            ctx.rotate(angle * Math.PI / 180);
            ctx.drawImage(img, -img.width / 2, -img.height / 2);
        });
    };

    const handleFlip = (direction: 'horizontal' | 'vertical') => {
        applyDestructiveTransform((canvas, ctx, img) => {
            canvas.width = img.width;
            canvas.height = img.height;
            if (direction === 'horizontal') {
                ctx.translate(img.width, 0);
                ctx.scale(-1, 1);
            } else {
                ctx.translate(0, img.height);
                ctx.scale(1, -1);
            }
            ctx.drawImage(img, 0, 0);
        });
    };

    const handleApplyStraighten = () => {
        if (straightenAngle === 0) return;
        const angle = straightenAngle;
        applyDestructiveTransform((canvas, ctx, img) => {
            const rad = angle * Math.PI / 180;
            const w = img.width;
            const h = img.height;
            const abs_cos = Math.abs(Math.cos(rad));
            const abs_sin = Math.abs(Math.sin(rad));
            const newW = Math.ceil(w * abs_cos + h * abs_sin);
            const newH = Math.ceil(w * abs_sin + h * abs_cos);
            canvas.width = newW;
            canvas.height = newH;
            ctx.translate(newW / 2, newH / 2);
            ctx.rotate(rad);
            ctx.drawImage(img, -w / 2, -h / 2);
        });
        setStraightenAngle(0);
    };
    
    const aspectRatioValue = useMemo(() => {
        if (aspectRatio === 'Free' || !canvasRef.current) return null;
        if (aspectRatio === 'Original') return canvasRef.current.width / canvasRef.current.height;
        if (aspectRatio === 'Custom') {
            const match = customAspectRatioString.match(/^(\d+(?:\.\d+)?):(\d+(?:\.\d+)?)$/);
            if (match) {
                const [w, h] = [parseFloat(match[1]), parseFloat(match[2])];
                if (h > 0) return w / h;
            }
            return null;
        }
        const presetMatch = aspectRatio.match(/^(\d+):(\d+)$/);
        if (presetMatch) return parseInt(presetMatch[1]) / parseInt(presetMatch[2]);
        return null;
    }, [aspectRatio, customAspectRatioString, canvasRef.current?.width]);

    const setCropToAspectRatio = (ar: string) => {
        setAspectRatio(ar);
        setCustomAspectRatioString('');
        if (!canvasRef.current) return;
        const { width, height } = canvasRef.current;
        let newWidth = width, newHeight = height;

        const numericAR = ar === '1:1' ? 1 : ar === '4:3' ? 4/3 : ar === '3:2' ? 3/2 : ar === '16:9' ? 16/9 : ar === '3:4' ? 3/4 : ar === '2:3' ? 2/3 : ar === '9:16' ? 9/16 : null;

        if (numericAR) {
            if (width / height > numericAR) newWidth = height * numericAR;
            else newHeight = width / numericAR;
        }
        setCropBox({ x: (width - newWidth) / 2, y: (height - newHeight) / 2, width: newWidth, height: newHeight });
    };
    
    const handleCustomAspectRatioKeydown = (e: React.KeyboardEvent) => {
        if (e.key === 'Enter') {
            setAspectRatio('Custom');
             if (!canvasRef.current || !aspectRatioValue) return;
            const { width, height } = canvasRef.current;
            let newWidth = width, newHeight = height;
            if (width / height > aspectRatioValue) newWidth = height * aspectRatioValue;
            else newHeight = width / aspectRatioValue;
            setCropBox({ x: (width - newWidth) / 2, y: (height - newHeight) / 2, width: newWidth, height: newHeight });
        }
    }

    const handleEnterCropMode = () => {
        if (!canvasRef.current) return;
        setIsCropping(true);
        setStraightenAngle(0);
        setAspectRatio('Original');
        setCropBox({ x: 0, y: 0, width: canvasRef.current.width, height: canvasRef.current.height });
        setCompositionOverlay('thirds');
    };

    const handleExitCropMode = () => {
        setIsCropping(false);
        setCropBox(null);
        setStraightenAngle(0);
        if (compositionOverlay === 'thirds') setCompositionOverlay('none');
    };
    
    const handleApplyCrop = () => {
        if (!cropBox) return;
        const newCropBox = {...cropBox}; // Make a copy
        applyDestructiveTransform((canvas, ctx, img) => {
            canvas.width = newCropBox.width;
            canvas.height = newCropBox.height;
            ctx.drawImage(img, newCropBox.x, newCropBox.y, newCropBox.width, newCropBox.height, 0, 0, newCropBox.width, newCropBox.height);
        });
        handleExitCropMode();
    };

    const handleCropMouseDown = (e: React.MouseEvent, handle: string) => {
        e.stopPropagation();
        const point = getPointOnImage(e.clientX, e.clientY);
        if (!point || !cropBox) return;
        cropDragInfo.current = { handle, startX: point.x, startY: point.y, initialRect: { ...cropBox } };
        setIsDraggingCrop(true);
    };
    
    const handleCropMouseMove = useCallback((e: MouseEvent) => {
        if (!cropDragInfo.current || !canvasRef.current) return;
        e.preventDefault();
        const point = getPointOnImage(e.clientX, e.clientY);
        if (!point) return;

        const { initialRect, handle } = cropDragInfo.current;
        const imageWidth = canvasRef.current.width, imageHeight = canvasRef.current.height;
        let newRect = { ...initialRect };
        const dx = point.x - cropDragInfo.current.startX, dy = point.y - cropDragInfo.current.startY;

        if (handle === 'body') { newRect.x += dx; newRect.y += dy; }
        else {
            let { x, y, width, height } = initialRect;
            if (handle.includes('r')) width += dx;
            if (handle.includes('l')) { x += dx; width -= dx; }
            if (handle.includes('b')) height += dy;
            if (handle.includes('t')) { y += dy; height -= dy; }
            if (aspectRatioValue) {
                const isCorner = handle.length === 2;
                if (isCorner) {
                    const newProposedWidth = handle.includes('l') ? initialRect.width - dx : initialRect.width + dx;
                    width = newProposedWidth;
                    height = width / aspectRatioValue;
                    if (handle.includes('t')) y = initialRect.y + initialRect.height - height;
                    if (handle.includes('l')) x = initialRect.x + initialRect.width - width;
                } else if (handle.includes('l') || handle.includes('r')) {
                    const newH = width / aspectRatioValue;
                    y += (height - newH) / 2; height = newH;
                } else {
                    const newW = height * aspectRatioValue;
                    x += (width - newW) / 2; width = newW;
                }
            }
            newRect = { x, y, width, height };
        }
        if (newRect.width < 20) newRect.width = 20;
        if (newRect.height < 20) newRect.height = 20;
        newRect.x = Math.max(0, newRect.x);
        newRect.y = Math.max(0, newRect.y);
        if (newRect.x + newRect.width > imageWidth) {
             if (handle.includes('l') || handle === 'body') newRect.x = imageWidth - newRect.width;
             else newRect.width = imageWidth - newRect.x;
        }
        if (newRect.y + newRect.height > imageHeight) {
            if (handle.includes('t') || handle === 'body') newRect.y = imageHeight - newRect.height;
            else newRect.height = imageHeight - newRect.y;
        }
        setCropBox(newRect);
    }, [getPointOnImage, aspectRatioValue]);
    
    const handleCropMouseUp = useCallback(() => {
        cropDragInfo.current = null;
        setIsDraggingCrop(false);
    }, []);

    useEffect(() => {
        if (isDraggingCrop) {
            window.addEventListener('mousemove', handleCropMouseMove);
            window.addEventListener('mouseup', handleCropMouseUp);
        }
        return () => {
            window.removeEventListener('mousemove', handleCropMouseMove);
            window.removeEventListener('mouseup', handleCropMouseUp);
        };
    }, [isDraggingCrop, handleCropMouseMove, handleCropMouseUp]);

    const liveStraightenTransform = useMemo(() => {
        const canvas = canvasRef.current;
        if (!canvas || !canvas.width || !canvas.height) {
            return `rotate(0deg) scale(1)`;
        }

        if (straightenAngle === 0) {
            return `rotate(0deg) scale(1)`;
        }

        const w = canvas.width;
        const h = canvas.height;
        const rad = Math.abs(straightenAngle * Math.PI / 180);

        const sin = Math.sin(rad);
        const cos = Math.cos(rad);

        // Calculate how much we need to scale to cover the original image dimensions
        const scale = Math.max((w * cos + h * sin) / w, (w * sin + h * cos) / h);

        return `rotate(${straightenAngle}deg) scale(${scale})`;
    }, [straightenAngle, canvasRef.current?.width, canvasRef.current?.height]);

    const CompositionOverlay = ({ type, width, height }) => {
      if (type === 'none' || !width || !height) return null;
      return <svg className="composition-overlay" viewBox={`0 0 ${width} ${height}`}>
          {type === 'thirds' && <path d={`M ${width/3} 0 L ${width/3} ${height} M ${width*2/3} 0 L ${width*2/3} ${height} M 0 ${height/3} L ${width} ${height/3} M 0 ${height*2/3} L ${width} ${height*2/3}`} />}
          {type === 'spiral' && <path d="M 124.9,94.3 C 124.9,111.4 111.4,124.9 94.3,124.9 C 77.2,124.9 63.7,111.4 63.7,94.3 C 63.7,77.2 77.2,63.7 94.3,63.7 C 111.4,63.7 124.9,77.2 124.9,94.3 Z M 94.3,116.3 C 106.3,116.3 116.3,106.3 116.3,94.3 C 116.3,82.3 106.3,72.3 94.3,72.3 C 82.3,72.3 72.3,82.3 72.3,94.3 C 72.3,106.3 82.3,116.3 94.3,116.3 Z M 116.3,94.3 C 116.3,94.3 116.3,94.3 116.3,94.3 L 116.3,94.3 C 116.3,94.3 116.3,94.3 116.3,94.3 Z M 94.3,124.9 C 85.8,124.9 78,121.5 72.3,116.3 C 66.5,121.5 58.7,124.9 50.2,124.9 C 22.5,124.9 0.1,102.5 0.1,74.8 C 0.1,47.1 22.5,24.7 50.2,24.7 C 77.9,24.7 100.3,47.1 100.3,74.8 C 100.3,80.9 98.9,86.8 96.5,92.1 M 50.2,33.3 C 27.2,33.3 8.7,51.8 8.7,74.8 C 8.7,97.8 27.2,116.3 50.2,116.3 C 57.5,116.3 64.3,114.1 70,110.4" transform={`translate(${width/2 - 100}, ${height/2 - 100}) scale(${Math.min(width, height) / 200})`} />}
      </svg>;
    };

    const CropUI = ({ rect, imageWidth, imageHeight, onMouseDown }) => {
        const handles = [
          { name: 'tl', cursor: 'nwse-resize', x: rect.x, y: rect.y },
          { name: 'tm', cursor: 'ns-resize', x: rect.x + rect.width / 2, y: rect.y },
          { name: 'tr', cursor: 'nesw-resize', x: rect.x + rect.width, y: rect.y },
          { name: 'ml', cursor: 'ew-resize', x: rect.x, y: rect.y + rect.height / 2 },
          { name: 'mr', cursor: 'ew-resize', x: rect.x + rect.width, y: rect.y + rect.height / 2 },
          { name: 'bl', cursor: 'nesw-resize', x: rect.x, y: rect.y + rect.height },
          { name: 'bm', cursor: 'ns-resize', x: rect.x + rect.width / 2, y: rect.y + rect.height },
          { name: 'br', cursor: 'nwse-resize', x: rect.x + rect.width, y: rect.y + rect.height },
        ];
        return <svg className="crop-overlay" viewBox={`0 0 ${imageWidth} ${imageHeight}`} onMouseDown={e => onMouseDown(e, 'body')}>
            <defs><mask id="cropMask"><rect x="0" y="0" width={imageWidth} height={imageHeight} fill="white"/><rect x={rect.x} y={rect.y} width={rect.width} height={rect.height} fill="black"/></mask></defs>
            <rect className="crop-scrim" x="0" y="0" width={imageWidth} height={imageHeight} mask="url(#cropMask)" />
            <rect className="crop-box" x={rect.x} y={rect.y} width={rect.width} height={rect.height} />
            {handles.map(h => <rect key={h.name} className="crop-handle" x={h.x - 5} y={h.y - 5} width="10" height="10" style={{ cursor: h.cursor }} onMouseDown={e => onMouseDown(e, h.name)} />)}
        </svg>;
    };
    
    const addLayer = (name?: string): string => {
        const newLayer: Layer = {
            id: `layer_${Date.now()}`,
            name: name || `Mask ${currentState.layers.length + 1}`,
            isVisible: true,
            maskSrc: null,
            isMaskInverted: false,
        };
        commitChange(draft => ({...draft, layers: [...draft.layers, newLayer]}));
        setActiveLayerId(newLayer.id);
        return newLayer.id;
    };

    const toggleLayerVisibility = (id: string) => {
        commitChange(draft => ({
            ...draft,
            layers: draft.layers.map(l => l.id === id ? { ...l, isVisible: !l.isVisible } : l)
        }));
    };

    const deleteLayer = (id: string) => {
        commitChange(draft => ({...draft, layers: draft.layers.filter(l => l.id !== id)}));
        if (activeLayerId === id) setActiveLayerId(null);
    };
    
    const updateLayerName = (id: string, newName: string) => {
        commitChange(draft => ({
            ...draft,
            layers: draft.layers.map(l => l.id === id ? { ...l, name: newName } : l)
        }));
    };
    
    const LayersPanel = () => (
        <div className="layers-panel">
            <div className="layer-list">
                {currentState.layers.map(layer => (
                    <div key={layer.id} className={`layer-item ${layer.id === activeLayerId ? 'active' : ''}`} onClick={() => setActiveLayerId(layer.id)}>
                        <button className="layer-visibility" onClick={(e) => { e.stopPropagation(); toggleLayerVisibility(layer.id); }}>
                            <span className="material-symbols-outlined">{layer.isVisible ? 'visibility' : 'visibility_off'}</span>
                        </button>
                        {renamingLayerId === layer.id ? (
                            <input
                                type="text"
                                defaultValue={layer.name}
                                autoFocus
                                onBlur={(e) => { updateLayerName(layer.id, e.currentTarget.value); setRenamingLayerId(null); }}
                                onKeyDown={(e) => { if (e.key === 'Enter') { updateLayerName(layer.id, e.currentTarget.value); setRenamingLayerId(null); } }}
                                onClick={e => e.stopPropagation()}
                            />
                        ) : (
                            <span className="layer-name" onDoubleClick={() => setRenamingLayerId(layer.id)}>{layer.name}</span>
                        )}
                        <button className="delete-layer-btn" onClick={(e) => { e.stopPropagation(); deleteLayer(layer.id); }}>
                             <span className="material-symbols-outlined">delete</span>
                        </button>
                    </div>
                ))}
                {currentState.layers.length === 0 && <p className="no-presets">No masks yet. Add one to get started.</p>}
            </div>
            <div className="layer-actions">
                <button className="btn btn-secondary" onClick={() => addLayer()} disabled={!originalImageSrc}>
                    <span className="material-symbols-outlined">add</span>Add Mask
                </button>
            </div>
        </div>
    );

  return (
    <div className={`app-wrapper ${isFullscreen ? 'fullscreen-active' : ''}`} ref={appWrapperRef}>
      {isLoading && <div className="loading-overlay"><MorphingLoader /><p>{loadingMessage}</p></div>}
      {showDownloadModal && <div className="modal-overlay" onClick={() => setShowDownloadModal(false)}>
          <div className="modal-content" onClick={e => e.stopPropagation()}>
              <h3>Export Image</h3>
              <div className="download-options">
                <label>Format:<select value={downloadOptions.format} onChange={e => setDownloadOptions({...downloadOptions, format: e.target.value as any})}><option value="image/png">PNG</option><option value="image/jpeg">JPEG</option></select></label>
                {downloadOptions.format === 'image/jpeg' && <WavySlider label="Quality" unit="%" value={downloadOptions.quality} min={1} max={100} onChange={value => setDownloadOptions({...downloadOptions, quality: value})} onDragEnd={() => {}} disabled={false} />}
              </div>
              <button className="btn btn-primary" onClick={exportImage}>Download</button>
          </div>
      </div>}
      {showGenFillModal && <div className="modal-overlay" onClick={() => setShowGenFillModal(false)}>
          <div className="modal-content" onClick={e => e.stopPropagation()}>
              <h3>{brushMode === 'sketch' ? 'Sketch to Image' : 'Generative Fill'}</h3>
              <p>{brushMode === 'sketch' ? "Describe what your drawing should become." : "Describe what you want to generate in the masked area."}</p>
              <input type="text" className="text-input" placeholder="e.g., 'A majestic castle'" value={brushMode === 'sketch' ? sketchPrompt : genFillPrompt} onChange={e => brushMode === 'sketch' ? setSketchPrompt(e.target.value) : setGenFillPrompt(e.target.value)} />
              <button className="btn btn-primary" onClick={() => applyAiAction(brushMode === 'sketch' ? 'sketch-to-image' : 'generative-fill', brushMode === 'sketch' ? sketchPrompt : genFillPrompt)}>Generate</button>
          </div>
      </div>}
      {showBetaMessage && <div className="modal-overlay" onClick={() => {setShowBetaMessage(false); localStorage.setItem('hasSeenBetaMessage', 'true');}}>
          <div className="modal-content" onClick={e => e.stopPropagation()}>
              <h3>Welcome!</h3>
              <p>This is a powerful, mask-based, AI-enhanced photo editor. Enjoy exploring!</p>
              <button className="btn btn-primary" onClick={() => {setShowBetaMessage(false); localStorage.setItem('hasSeenBetaMessage', 'true');}}>Get Started</button>
          </div>
      </div>}
      <div className="app-container">
        <div className="main-content">
          <div className={`toolbar ${isMobilePanelCollapsed ? 'collapsed' : ''}`}>
               <div className="mobile-toolbar-handle" onClick={() => setIsMobilePanelCollapsed(s => !s)}>
                  <span className="material-symbols-outlined">
                      {isMobilePanelCollapsed ? 'keyboard_arrow_up' : 'keyboard_arrow_down'}
                  </span>
              </div>
              <div className="top-actions">
                  <button className={`presets-main-button ${activeTab === 'presets' ? 'active' : ''}`} disabled={!originalImageSrc} onClick={() => handleTabClick('presets')}>Presets</button>
                  <button className="save-main-button" disabled={!originalImageSrc} onClick={savePreset} aria-label="Save current adjustments as preset"><span className="material-symbols-outlined">library_add</span></button>
              </div>
              
              <div className="panel-container">
                  {activeTab !== 'presets' ? (
                      <div className="adjustments-panel animated-panel">
                          <div className="sub-nav" ref={subNavRef}>
                              {TABS.filter(t => t.id !== 'presets').map(tab => (
                                  <button key={tab.id} ref={el => { subNavButtonRefs.current.set(tab.id, el); }} className={`sub-nav-button ${activeTab === tab.id ? 'active' : ''}`} disabled={!originalImageSrc} onClick={() => handleTabClick(tab.id)} onMouseDown={handleInteractionAnimation} title={tab.title}>
                                      {typeof tab.icon === 'string' ? <span className="material-symbols-outlined">{tab.icon}</span> : tab.icon}
                                  </button>
                              ))}
                              <div className="active-tab-indicator" style={{...indicatorStyle}}></div>
                          </div>
                          <div className="panel-content-inner">
                              <div key={activeTab} className="animated-content">
                                  {activeTab && TABS.find(t => t.id === activeTab) && (<p className="panel-content-heading">{TABS.find(t => t.id === activeTab)?.title}</p>)}
                                  <div className="panel-content">
                                      {activeTab === 'masking' && (<>
                                        <div className="tool-section"><h4>Brush Controls</h4><div className="brush-controls-group"><div className="button-row">
                                            <button title="Masking Brush" className={`btn-tool ${brushMode === 'mask' ? 'active' : ''}`} onClick={() => setBrushMode(brushMode === 'mask' ? 'none' : 'mask')} disabled={!activeLayer}><span className="material-symbols-outlined">brush</span></button>
                                            <button title="Eraser" className={`btn-tool ${brushMode === 'eraser' ? 'active' : ''}`} onClick={() => setBrushMode(brushMode === 'eraser' ? 'none' : 'eraser')} disabled={!activeLayer}><span className="material-symbols-outlined">ink_eraser</span></button>
                                            </div><WavySlider label="Size" unit="px" value={brushSize} min={1} max={200} onChange={value => setBrushSize(value)} onDragEnd={() => {}} disabled={brushMode === 'none'} /></div>
                                        </div>
                                        <div className="tool-section">
                                            <h4>AI Selection</h4>
                                            <div className="button-row">
                                                <button title="Select Subject" className="btn-tool" onClick={() => handleGenerateAiMask('subject')} disabled={!originalImageSrc || isLoading}>
                                                    <span className="material-symbols-outlined">person</span>
                                                </button>
                                                <button title="Select Sky" className="btn-tool" onClick={() => handleGenerateAiMask('sky')} disabled={!originalImageSrc || isLoading}>
                                                    <span className="material-symbols-outlined">filter_drama</span>
                                                </button>
                                            </div>
                                        </div>
                                        <div className="tool-section">
                                            <h4>Gradients</h4>
                                            <div className="button-row">
                                                <button title="Linear Gradient" className={`btn-tool ${brushMode === 'linear-gradient' ? 'active' : ''}`} onClick={() => setBrushMode(brushMode === 'linear-gradient' ? 'none' : 'linear-gradient')} disabled={!originalImageSrc}>
                                                    <span className="material-symbols-outlined">linear_scale</span>
                                                </button>
                                                <button title="Radial Gradient" className={`btn-tool ${brushMode === 'radial-gradient' ? 'active' : ''}`} onClick={() => setBrushMode(brushMode === 'radial-gradient' ? 'none' : 'radial-gradient')} disabled={!originalImageSrc}>
                                                    <span className="material-symbols-outlined">vignette</span>
                                                </button>
                                            </div>
                                        </div>
                                        <div className="tool-section"><h4>Mask Options</h4><div className="toggle-switch"><label>Invert Mask</label><label className="switch"><input type="checkbox" checked={activeLayer?.isMaskInverted ?? false} onChange={e => activeLayer && commitChange(draft => ({ ...draft, layers: draft.layers.map(l => l.id === activeLayerId ? { ...l, isMaskInverted: e.target.checked } : l)}))} disabled={!activeLayer?.maskSrc}/><span className="switch-slider"></span></label></div><button className="btn btn-secondary" onClick={clearActiveMask} disabled={!activeLayer?.maskSrc}>Clear Mask</button></div>
                                        <div className="tool-section"><h4>Generative AI</h4><button className="btn btn-secondary" onClick={() => setShowGenFillModal(true)} disabled={!activeLayer?.maskSrc}>Generative Fill</button><p className="tool-note">Use the brush to mask an area, then click 'Generative Fill' to replace it with something new.</p></div>
                                        <div className="tool-section"><h4>Masks</h4><LayersPanel /></div>
                                      </>)}
                                      {activeTab === 'transform' && ( <>
                                            <div className="tool-section"><h4>Aspect Ratio</h4><div className="aspect-ratio-presets">{['Original', 'Free', '1:1', '4:3', '16:9', '3:4', '9:16'].map(ar => (<button key={ar} className={`btn-aspect-ratio ${aspectRatio === ar ? 'active' : ''}`} onClick={() => setCropToAspectRatio(ar)} disabled={!isCropping}>{ar}</button>))}</div><input type="text" className="text-input" placeholder="Custom (e.g., 21:9)" value={customAspectRatioString} onChange={e => setCustomAspectRatioString(e.target.value)} onKeyDown={handleCustomAspectRatioKeydown} disabled={!isCropping} /></div>
                                            <div className="tool-section"><h4>Crop</h4><div className="button-group">{!isCropping ? (<button className="btn btn-secondary" onClick={handleEnterCropMode} disabled={!originalImageSrc}>Enter Crop Mode</button>) : (<><button className="btn btn-primary" onClick={handleApplyCrop} disabled={!originalImageSrc}>Apply Crop</button><button className="btn btn-secondary" onClick={handleExitCropMode} disabled={!originalImageSrc}>Cancel</button></>)}</div></div>
                                            <div className="tool-section"><h4>Transform</h4><div className="button-row"><button title="Rotate Left" className="btn-tool" onClick={() => handleRotate(-90)} disabled={!originalImageSrc || isCropping}><span className="material-symbols-outlined">rotate_90_degrees_ccw</span></button><button title="Rotate Right" className="btn-tool" onClick={() => handleRotate(90)} disabled={!originalImageSrc || isCropping}><span className="material-symbols-outlined">rotate_90_degrees_cw</span></button><button title="Flip Horizontal" className="btn-tool" onClick={() => handleFlip('horizontal')} disabled={!originalImageSrc || isCropping}><span className="material-symbols-outlined">flip</span></button><button title="Flip Vertical" className="btn-tool" onClick={() => handleFlip('vertical')} disabled={!originalImageSrc || isCropping}><span className="material-symbols-outlined" style={{transform: 'rotate(90deg)'}}>flip</span></button></div><WavySlider label="Straighten" unit="°" value={straightenAngle} min={-45} max={45} onChange={value => setStraightenAngle(value)} onDragEnd={() => {}} disabled={!originalImageSrc || isCropping} /><button className="btn btn-secondary" onClick={handleApplyStraighten} disabled={straightenAngle === 0 || !originalImageSrc}>Apply Straighten</button></div>
                                            <div className="tool-section"><h4>Composition Overlays</h4><div className="button-row"><button title="Rule of Thirds" className={`btn-tool ${compositionOverlay === 'thirds' ? 'active' : ''}`} onClick={() => setCompositionOverlay(o => o === 'thirds' ? 'none' : 'thirds')} disabled={!originalImageSrc}><span className="material-symbols-outlined">grid_on</span></button><button title="Golden Spiral" className={`btn-tool ${compositionOverlay === 'spiral' ? 'active' : ''}`} onClick={() => setCompositionOverlay(o => o === 'spiral' ? 'none' : 'spiral')} disabled={!originalImageSrc}><span className="material-symbols-outlined">interests</span></button></div></div>
                                      </>)}
                                      {activeTab === 'basic' && (<div className="slider-grid">
                                          <WavySlider label="Brightness" unit="%" value={activeAdjustments.filters.brightness} min={0} max={200} onChange={value => handleFilterChange('brightness', value)} onDragEnd={handleCommitUpdate} disabled={!originalImageSrc} />
                                          <WavySlider label="Contrast" unit="%" value={activeAdjustments.filters.contrast} min={0} max={200} onChange={value => handleFilterChange('contrast', value)} onDragEnd={handleCommitUpdate} disabled={!originalImageSrc} />
                                          <WavySlider label="Saturation" unit="%" value={activeAdjustments.filters.saturation} min={0} max={200} onChange={value => handleFilterChange('saturation', value)} onDragEnd={handleCommitUpdate} disabled={!originalImageSrc} />
                                          <CommitSlider label="Exposure" unit="" value={activeAdjustments.filters.exposure} min={-100} max={100} onCommit={value => handleFilterCommit('exposure', value)} disabled={!originalImageSrc} />
                                          <CommitSlider label="Highlights" unit="" value={activeAdjustments.filters.highlights} min={-100} max={100} onCommit={value => handleFilterCommit('highlights', value)} disabled={!originalImageSrc} />
                                          <CommitSlider label="Shadows" unit="" value={activeAdjustments.filters.shadows} min={-100} max={100} onCommit={value => handleFilterCommit('shadows', value)} disabled={!originalImageSrc} />
                                          <CommitSlider label="Whites" unit="" value={activeAdjustments.filters.whites} min={-100} max={100} onCommit={value => handleFilterCommit('whites', value)} disabled={!originalImageSrc} />
                                          <CommitSlider label="Blacks" unit="" value={activeAdjustments.filters.blacks} min={-100} max={100} onCommit={value => handleFilterCommit('blacks', value)} disabled={!originalImageSrc} />
                                          <CommitSlider label="Temperature" unit="" value={activeAdjustments.filters.temperature} min={-100} max={100} onCommit={value => handleFilterCommit('temperature', value)} disabled={!originalImageSrc} />
                                          <CommitSlider label="Vibrance" unit="" value={activeAdjustments.filters.vibrance} min={-100} max={100} onCommit={value => handleFilterCommit('vibrance', value)} disabled={!originalImageSrc} />
                                      </div>)}
                                      {activeTab === 'curves' && <CurvesEditor curves={activeAdjustments.curves} onCurveChange={handleCurveChange} onCommit={handleCommitUpdate} disabled={!originalImageSrc} histogram={histogram} />}
                                      {activeTab === 'color' && (<div className="hsl-sections-container">{Object.keys(HSL_RANGES).map(color => (<div key={color} className="hsl-color-section">
                                        <h4 className="hsl-color-label" style={{color: color}}>{color}</h4>
                                        <CommitSlider label="Hue" unit="" value={activeAdjustments.hsl[color as keyof HSLFilters]?.h ?? 0} min={-100} max={100} onCommit={value => handleHslCommit(color as keyof HSLFilters, 'h', value)} disabled={!originalImageSrc} />
                                        <CommitSlider label="Saturation" unit="" value={activeAdjustments.hsl[color as keyof HSLFilters]?.s ?? 0} min={-100} max={100} onCommit={value => handleHslCommit(color as keyof HSLFilters, 's', value)} disabled={!originalImageSrc} />
                                        <CommitSlider label="Luminance" unit="" value={activeAdjustments.hsl[color as keyof HSLFilters]?.l ?? 0} min={-100} max={100} onCommit={value => handleHslCommit(color as keyof HSLFilters, 'l', value)} disabled={!originalImageSrc} />
                                      </div>))}</div>)}
                                      {activeTab === 'effects' && (<div className="slider-grid">
                                          <CommitSlider label="Sharpen" unit="%" value={activeAdjustments.filters.sharpen} min={0} max={100} onCommit={value => handleFilterCommit('sharpen', value)} disabled={!originalImageSrc} />
                                          <CommitSlider label="Grain" unit="%" value={activeAdjustments.filters.grain} min={0} max={100} onCommit={value => handleFilterCommit('grain', value)} disabled={!originalImageSrc} />
                                          <WavySlider label="Sepia" unit="%" value={activeAdjustments.filters.sepia} min={0} max={100} onChange={value => handleFilterChange('sepia', value)} onDragEnd={handleCommitUpdate} disabled={!originalImageSrc} />
                                          <CommitSlider label="Dehaze" unit="" value={activeAdjustments.filters.dehaze} min={-100} max={100} onCommit={value => handleFilterCommit('dehaze', value)} disabled={!originalImageSrc} />
                                          <CommitSlider label="Haze Amount" unit="%" value={activeAdjustments.filters.haze} min={0} max={100} onCommit={value => handleFilterCommit('haze', value)} disabled={!originalImageSrc} />
                                          <CommitSlider label="Haze Warmth" unit="" value={activeAdjustments.filters.hazeSpread} min={0} max={100} onCommit={value => handleFilterCommit('hazeSpread', value)} disabled={!originalImageSrc} />
                                      </div>)}
                                      {activeTab === 'ai-tools' && (<>
                                          <div className="tool-section"><div className="ai-studio-notice"><span className="material-symbols-outlined">sparkle</span><p>AI Studio tools analyze and modify your image.</p></div><div className="button-group"><button className="btn btn-secondary" onClick={() => applyAiAction('enhance')} disabled={!originalImageSrc}>Auto Enhance</button><button className="btn btn-secondary" onClick={() => applyAiAction('analyze-composition')} disabled={!originalImageSrc}>Analyze Composition</button></div></div>
                                          <div className="tool-section"><h4>Sketch to Image</h4><p className="tool-note">Draw a sketch on your image, describe it, and watch AI bring it to life.</p><div className="brush-controls-group"><div className="button-row"><button title="Sketch Brush" className={`btn-tool ${brushMode === 'sketch' ? 'active' : ''}`} onClick={() => setBrushMode(brushMode === 'sketch' ? 'none' : 'sketch')} disabled={!activeLayer}><span className="material-symbols-outlined">edit</span></button><button title="Clear Sketch" className="btn-tool" onClick={clearActiveMask} disabled={!activeLayer?.maskSrc}><span className="material-symbols-outlined">delete</span></button></div><WavySlider label="Size" unit="px" value={brushSize} min={1} max={200} onChange={value => setBrushSize(value)} onDragEnd={() => {}} disabled={brushMode !== 'sketch'} /></div>{brushMode === 'sketch' && (<div className="prompt-group"><input type="text" className="text-input" placeholder="e.g., 'a small boat on the water'" value={sketchPrompt} onChange={e => setSketchPrompt(e.target.value)} /><button className="btn btn-primary" onClick={() => applyAiAction('sketch-to-image', sketchPrompt)} disabled={!activeLayer?.maskSrc || !sketchPrompt}>Generate</button></div>)}</div>
                                          <div className="tool-section"><h4>Magic Remove</h4><p className="tool-note">Circle an object to remove it instantly. Best for small- to medium-sized objects.</p><div className="button-row"><button title="Magic Remove Brush" className={`btn-tool ${brushMode === 'magic-remove' ? 'active' : ''}`} onClick={() => setBrushMode(brushMode === 'magic-remove' ? 'none' : 'magic-remove')} disabled={!activeLayer}><span className="material-symbols-outlined">auto_fix_normal</span></button></div></div>
                                      </>)}
                                  </div>
                              </div>
                          </div>
                      </div>
                  ) : (
                      <div className="presets-panel animated-panel"><div key="presets" className="animated-content">
                          <div className="panel-content-wrapper presets-panel">
                              <div className="tool-section"><h4>Built-in Presets</h4><div className="preset-group"><span className="no-presets">Coming soon!</span></div></div>
                              <div className="tool-section"><h4>My Presets</h4>{customPresets.length > 0 ? (<div className="custom-presets-list">{customPresets.map((p, i) => (<div key={i} className="custom-preset-item"><span onClick={() => applyPreset(p)}>{p.name}</span><button className="delete-preset-btn" onClick={() => deletePreset(i)}><span className="material-symbols-outlined">delete</span></button></div>))}</div>) : (<p className="no-presets">You haven't saved any presets yet.</p>)}</div>
                          </div>
                      </div></div>
                  )}
              </div>
          </div>
          <div className="canvas-area">
            <div className="canvas-header"><div className="header-controls">
                {compositionAdvice && <div className="ai-studio-notice" style={{padding: '0.5rem 1rem', cursor: 'pointer'}} onClick={() => setCompositionAdvice(null)}><p>{compositionAdvice}</p></div>}
                <button className="header-btn-circle" onClick={() => setShowDownloadModal(true)} onMouseDown={handleInteractionAnimation} disabled={!originalImageSrc} title="Export Image"><span className="material-symbols-outlined">upload</span></button>
                <button className="header-btn-square" onClick={() => setShowMenu(s => !s)} onMouseDown={handleInteractionAnimation} title="Menu"><span className="material-symbols-outlined">more_vert</span></button>
                {showMenu && <div className="menu-dropdown"><button onClick={() => fileInputRef.current?.click()}><span className="material-symbols-outlined">download</span> Upload Image</button><button onClick={resetAll} disabled={!originalImageSrc}><span className="material-symbols-outlined">refresh</span> Reset Edits</button></div>}
                <button className="header-btn-circle" onClick={() => setShowThemes(s => !s)} onMouseDown={handleInteractionAnimation} title="Change Theme"><span className="material-symbols-outlined">palette</span></button>
                {showThemes && <div className="themes-dropdown"><button onClick={() => handleThemeChange('light')}>Light</button><button onClick={() => handleThemeChange('dark')}>Dark</button><button onClick={() => handleThemeChange('mono-light')}>Mono Light</button><button onClick={() => handleThemeChange('mono-dark')}>Mono Dark</button></div>}
            </div></div>
            <input type="file" ref={fileInputRef} onChange={handleFileChange} accept="image/*" style={{ display: 'none' }} />
            <div className="canvas-container" ref={canvasWrapperRef} onWheel={handleZoom} onClick={() => { setShowMenu(false); setShowThemes(false); }}>
                {isProcessing && <div className="processing-indicator"><MorphingLoader /></div>}
                {originalImageSrc ? (
                    <div className={`canvas-wrapper ${isPeeking ? 'peeking' : ''} ${brushMode !== 'none' || isCropping ? 'brush-active' : ''}`} style={{ transform: `scale(${transform.zoom}) translateX(${transform.pan.x / transform.zoom}px) translateY(${transform.pan.y / transform.zoom}px)`}} onMouseDown={handleCanvasMouseDown} onMouseMove={handleCanvasMouseMove} onMouseUp={handleCanvasMouseUp} onMouseLeave={handleCanvasMouseUp}>
                        <div className="image-rotator" style={{ transform: liveStraightenTransform }}>
                            <canvas ref={canvasRef} />
                            <canvas ref={maskCanvasRef} className="mask-canvas" />
                            {isCropping && cropBox && canvasRef.current && <CropUI rect={cropBox} imageWidth={canvasRef.current.width} imageHeight={canvasRef.current.height} onMouseDown={handleCropMouseDown} />}
                            {compositionOverlay !== 'none' && canvasRef.current && <CompositionOverlay type={compositionOverlay} width={canvasRef.current.width} height={canvasRef.current.height} />}
                        </div>
                        <img src={originalImageSrc} className="peek-image" style={{ width: canvasRef.current?.width, height: canvasRef.current?.height }} alt="Original"/>
                    </div>
                ) : (
                    <button className="placeholder" onClick={() => fileInputRef.current?.click()}>
                        <span className="material-symbols-outlined">add_photo_alternate</span><p>Upload a photo to start editing</p>
                    </button>
                )}
            </div>
            {originalImageSrc && (<div className="bottom-controls-bar">
                <div className="view-control-group zoom-group">
                    <button className="view-btn-round" onClick={() => zoomWithButtons('out')} disabled={!originalImageSrc}><span className="material-symbols-outlined">remove</span></button>
                    <div className="zoom-slider-container"><input type="range" className="view-zoom-slider" min="0.1" max="5" step="0.01" value={transform.zoom} onChange={e => handleSliderZoom(parseFloat(e.target.value))} disabled={!originalImageSrc} /></div>
                    <button className="view-btn-round" onClick={() => zoomWithButtons('in')} disabled={!originalImageSrc}><span className="material-symbols-outlined">add</span></button>
                </div>
                <div className="view-control-group">
                    <button className="view-btn-round" onClick={fitToScreen} disabled={!originalImageSrc} title="Fit to screen"><span className="material-symbols-outlined">fit_screen</span></button>
                    <button className="view-btn-round" onMouseDown={() => setIsPeeking(true)} onMouseUp={() => setIsPeeking(false)} onTouchStart={() => setIsPeeking(true)} onTouchEnd={() => setIsPeeking(false)} disabled={!originalImageSrc} title="Peek at original"><span className="material-symbols-outlined">visibility</span></button>
                    <button className="view-btn-round" onClick={toggleFullscreen} disabled={!originalImageSrc} title="Fullscreen">
                        <span className="material-symbols-outlined">{isFullscreen ? 'fullscreen_exit' : 'fullscreen'}</span>
                    </button>
                </div>
                <div className="view-control-group">
                    <button className="view-btn-round" onClick={undo} disabled={!canUndo} title="Undo"><span className="material-symbols-outlined">undo</span></button>
                    <button className="view-btn-round" onClick={redo} disabled={!canRedo} title="Redo"><span className="material-symbols-outlined">redo</span></button>
                </div>
            </div>)}
          </div>
        </div>
      </div>
    </div>
  );
};

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(<App />);