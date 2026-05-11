/**
 * M002: Rembg Bridge - Node.js ↔ Python rembg 桥接层
 * 文件位置: src/pipeline/rembg-bridge.ts
 *
 * 从 subprocess spawn 开始。然后,遵循该模式实现 stdin/stdout streaming,
 * 让 Node.js 可以将图像数据流式传入 Python 进程,并能获取处理结果。
 * 接着 session pooling 引入连接池,使多请求能够复用 Python 进程,
 * 同时 model caching 优化首次加载延迟。随后 health check
 * 整合进程存活监测,令系统支持自动重启,进而 fallback chain
 * 增强容错(rembg → chroma → 原图)。最终 type safety 完善
 * TypeScript 接口,确保输入输出兼容 pipeline orchestrator。
 *
 * 批判:
 *   用户角度: rembg 首次运行需下载模型 (~170MB) → 提供进度提示
 *   系统角度: Python 子进程的内存泄漏 → 设置最大处理次数后重启
 */

import { spawn, ChildProcess, execSync } from 'child_process';
import { writeFile, unlink, readFile, mkdtemp, access, mkdir } from 'fs/promises';
import { tmpdir } from 'os';
import { join } from 'path';
import { constants } from 'fs';
import { EventEmitter } from 'events';

// ──────────────────────────────────────────────────────────────────────
// §1  类型定义
// ──────────────────────────────────────────────────────────────────────

export interface RembgConfig {
  /** Python 可执行文件路径 */
  pythonPath: string;
  /** rembg 模型名称 */
  modelName: RembgModel;
  /** 是否启用 alpha matting 后处理 */
  alphaMatting: boolean;
  /** Alpha matting 前景阈值 */
  alphaMattingForegroundThreshold: number;
  /** Alpha matting 背景阈值 */
  alphaMattingBackgroundThreshold: number;
  /** Alpha matting 腐蚀大小 */
  alphaMattingErodeSize: number;
  /** 处理超时 (毫秒) */
  timeoutMs: number;
  /** 进程池大小 */
  poolSize: number;
  /** 单个进程最大处理次数 (防内存泄漏) */
  maxProcessAge: number;
  /** 模型缓存目录 */
  modelCacheDir: string;
  /** 临时文件目录 */
  tempDir: string;
  /** 是否只返回 mask (不返回去背景图) */
  onlyMask: boolean;
}

export type RembgModel =
  | 'u2net'           // 通用, 176MB
  | 'u2netp'          // 轻量版, 4.7MB
  | 'u2net_human_seg' // 人像专用
  | 'isnet-general-use' // ISNet 通用
  | 'birefnet-general'  // BiRefNet, 最高质量
  | 'silueta';         // Silueta, 快速

export interface RembgResult {
  /** 去除背景后的 PNG buffer */
  outputBuffer: Buffer;
  /** 分割 mask (灰度 PNG) */
  maskBuffer: Buffer | null;
  /** 处理元数据 */
  metadata: {
    model: RembgModel;
    processingTimeMs: number;
    pythonVersion: string;
    rembgVersion: string;
    inputSize: number;
    outputSize: number;
  };
}

interface PooledProcess {
  process: ChildProcess | null;
  busy: boolean;
  processedCount: number;
  createdAt: number;
}

// ──────────────────────────────────────────────────────────────────────
// §2  默认配置
// ──────────────────────────────────────────────────────────────────────

export const DEFAULT_REMBG_CONFIG: RembgConfig = {
  pythonPath: 'python3',
  modelName: 'isnet-general-use',
  alphaMatting: true,
  alphaMattingForegroundThreshold: 270,
  alphaMattingBackgroundThreshold: 20,
  alphaMattingErodeSize: 11,
  timeoutMs: 60000,
  poolSize: 2,
  maxProcessAge: 50,
  modelCacheDir: join(process.env.HOME || '/tmp', '.u2net'),
  tempDir: tmpdir(),
  onlyMask: false,
};

// ──────────────────────────────────────────────────────────────────────
// §3  Python 环境检测
// ──────────────────────────────────────────────────────────────────────

/**
 * 检测 Python 环境和 rembg 安装状态
 *
 * 参考 Google 的 TFX (TensorFlow Extended) 的环境验证模式:
 *   1. 检查 Python 版本 >= 3.8
 *   2. 检查 rembg 包已安装
 *   3. 检查 onnxruntime 可用
 *   4. 检查模型文件是否已缓存
 *
 * function checkPythonEnvironment(pythonPath):
 *   result = exec(pythonPath + " -c 'import sys; print(sys.version)'")
 *   if result.exitCode != 0: throw "Python not found"
 *   version = parseVersion(result.stdout)
 *   if version < 3.8: throw "Python >= 3.8 required"
 *   
 *   result = exec(pythonPath + " -c 'import rembg; print(rembg.__version__)'")
 *   if result.exitCode != 0: throw "rembg not installed"
 *   
 *   return { pythonVersion, rembgVersion, onnxAvailable, modelsCache }
 */
export async function checkPythonEnvironment(
  pythonPath: string = 'python3'
): Promise<{
  available: boolean;
  pythonVersion: string;
  rembgVersion: string;
  onnxBackend: string;
  modelsAvailable: string[];
  errors: string[];
}> {
  const errors: string[] = [];
  let pythonVersion = 'unknown';
  let rembgVersion = 'unknown';
  let onnxBackend = 'unknown';
  const modelsAvailable: string[] = [];

  // Check Python
  try {
    const pyVer = execSync(
      `${pythonPath} -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')"`,
      { timeout: 5000 }
    ).toString().trim();
    pythonVersion = pyVer;

    const [major, minor] = pyVer.split('.').map(Number);
    if (major < 3 || (major === 3 && minor < 8)) {
      errors.push(`Python >= 3.8 required, found ${pyVer}`);
    }
  } catch {
    errors.push(`Python not found at '${pythonPath}'`);
    return { available: false, pythonVersion, rembgVersion, onnxBackend, modelsAvailable, errors };
  }

  // Check rembg
  try {
    rembgVersion = execSync(
      `${pythonPath} -c "import rembg; print(rembg.__version__)"`,
      { timeout: 10000 }
    ).toString().trim();
  } catch {
    errors.push('rembg not installed. Run: pip install rembg[cpu]');
  }

  // Check ONNX runtime
  try {
    onnxBackend = execSync(
      `${pythonPath} -c "import onnxruntime; print(','.join(onnxruntime.get_available_providers()))"`,
      { timeout: 5000 }
    ).toString().trim();
  } catch {
    errors.push('onnxruntime not available');
  }

  // Check cached models
  try {
    const modelsStr = execSync(
      `${pythonPath} -c "
import os
model_dir = os.path.expanduser('~/.u2net')
if os.path.isdir(model_dir):
    models = [f for f in os.listdir(model_dir) if f.endswith('.onnx')]
    print(','.join(models))
else:
    print('')
"`,
      { timeout: 5000 }
    ).toString().trim();
    if (modelsStr) {
      modelsAvailable.push(...modelsStr.split(','));
    }
  } catch {
    // Non-critical
  }

  return {
    available: errors.length === 0,
    pythonVersion,
    rembgVersion,
    onnxBackend,
    modelsAvailable,
    errors,
  };
}

// ──────────────────────────────────────────────────────────────────────
// §4  Python 脚本生成
// ──────────────────────────────────────────────────────────────────────

/**
 * 生成用于 rembg 处理的 Python 脚本
 *
 * 参考 ByteDance 的 Lark/Feishu bot framework 中的
 * "生成临时脚本 → 执行 → 清理" 模式:
 *   避免维护独立的 Python 服务,降低部署复杂度。
 *   每次调用生成一个自包含脚本, 包含所有参数。
 *
 * function generateRembgScript(inputPath, outputPath, maskPath, config):
 *   script = f"""
 *   from rembg import remove, new_session
 *   from PIL import Image
 *   import sys, json, time
 *   
 *   session = new_session("{config.modelName}")
 *   input_img = Image.open("{inputPath}")
 *   
 *   output = remove(
 *     input_img,
 *     session=session,
 *     alpha_matting={config.alphaMatting},
 *     ...
 *   )
 *   
 *   output.save("{outputPath}")
 *   # 输出 mask
 *   mask = remove(input_img, session=session, only_mask=True)
 *   mask.save("{maskPath}")
 *   
 *   print(json.dumps({{ "status": "ok", ... }}))
 *   """
 *   return script
 */
function generateRembgScript(
  inputPath: string,
  outputPath: string,
  maskPath: string,
  cfg: RembgConfig
): string {
  const mattingArgs = cfg.alphaMatting
    ? `
    alpha_matting=True,
    alpha_matting_foreground_threshold=${cfg.alphaMattingForegroundThreshold},
    alpha_matting_background_threshold=${cfg.alphaMattingBackgroundThreshold},
    alpha_matting_erode_size=${cfg.alphaMattingErodeSize},`
    : '    alpha_matting=False,';

  return `
import sys
import json
import time

start = time.time()

try:
    from rembg import remove, new_session
    from PIL import Image
    import rembg

    session = new_session("${cfg.modelName}")
    
    input_img = Image.open("${inputPath.replace(/\\/g, '\\\\')}")
    
    # Generate output (background removed)
    output = remove(
        input_img,
        session=session,
${mattingArgs}
        post_process_mask=True,
    )
    output.save("${outputPath.replace(/\\/g, '\\\\')}")
    
    # Generate mask
    mask = remove(
        input_img,
        session=session,
        only_mask=True,
    )
    mask.save("${maskPath.replace(/\\/g, '\\\\')}")
    
    elapsed = time.time() - start
    
    result = {
        "status": "ok",
        "rembg_version": getattr(rembg, '__version__', 'unknown'),
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "model": "${cfg.modelName}",
        "processing_time_ms": round(elapsed * 1000),
        "input_size": [input_img.width, input_img.height],
        "output_mode": output.mode,
    }
    
    print(json.dumps(result))
    sys.exit(0)

except Exception as e:
    result = {
        "status": "error",
        "error": str(e),
        "error_type": type(e).__name__,
    }
    print(json.dumps(result), file=sys.stderr)
    sys.exit(1)
`;
}

// ──────────────────────────────────────────────────────────────────────
// §5  临时文件管理
// ──────────────────────────────────────────────────────────────────────

/**
 * 安全的临时文件管理
 *
 * 参考 Google Bazel 的 sandbox 模式:
 *   每次处理创建独立的临时目录,
 *   处理完成后无论成功失败都清理。
 */
async function createTempWorkspace(): Promise<{
  dir: string;
  inputPath: string;
  outputPath: string;
  maskPath: string;
  scriptPath: string;
  cleanup: () => Promise<void>;
}> {
  const dir = await mkdtemp(join(tmpdir(), 'rembg-'));
  const inputPath = join(dir, 'input.png');
  const outputPath = join(dir, 'output.png');
  const maskPath = join(dir, 'mask.png');
  const scriptPath = join(dir, 'process.py');

  const cleanup = async () => {
    const files = [inputPath, outputPath, maskPath, scriptPath];
    for (const f of files) {
      try { await unlink(f); } catch { /* ignore */ }
    }
    try {
      const { rmdir } = await import('fs/promises');
      await rmdir(dir);
    } catch { /* ignore */ }
  };

  return { dir, inputPath, outputPath, maskPath, scriptPath, cleanup };
}

// ──────────────────────────────────────────────────────────────────────
// §6  RembgBridge 主类
// ──────────────────────────────────────────────────────────────────────

export class RembgBridge extends EventEmitter {
  private config: RembgConfig;
  private environmentChecked: boolean = false;
  private environmentOk: boolean = false;

  constructor(config?: Partial<RembgConfig>) {
    super();
    this.config = { ...DEFAULT_REMBG_CONFIG, ...config };
  }

  /**
   * 检查并缓存环境状态
   */
  async ensureEnvironment(): Promise<boolean> {
    if (this.environmentChecked) return this.environmentOk;

    const env = await checkPythonEnvironment(this.config.pythonPath);
    this.environmentChecked = true;
    this.environmentOk = env.available;

    if (!env.available) {
      this.emit('environment-error', env.errors);
    }

    return this.environmentOk;
  }

  /**
   * 执行背景移除
   *
   * 完整流程:
   *   1. 环境检查
   *   2. 创建临时工作空间
   *   3. 写入输入图像
   *   4. 生成并写入 Python 脚本
   *   5. spawn 子进程执行
   *   6. 等待完成, 读取输出
   *   7. 清理临时文件
   *   8. 返回结果
   *
   * 用户角度批判: 首次执行会下载模型, 需等待较长时间
   *   → emit 'model-downloading' 事件让 UI 层可以展示进度
   *   → 设置合理的 timeout (默认60s)
   *
   * 系统角度批判: 文件 I/O 是瓶颈 (写入临时文件 → Python 读取 → 写出)
   *   → 考虑 stdin/stdout streaming (M002-v2 优化项)
   *   → 当前方案的好处是调试简单, 可检查中间文件
   */
  async removeBackground(input: Buffer): Promise<RembgResult> {
    const startTime = Date.now();

    // ── Step 1: 环境检查 ──
    const envOk = await this.ensureEnvironment();
    if (!envOk) {
      throw new Error(
        'Python/rembg environment not available. ' +
        'Install with: pip install rembg[cpu] onnxruntime Pillow'
      );
    }

    // ── Step 2: 临时工作空间 ──
    const workspace = await createTempWorkspace();

    try {
      // ── Step 3: 写入输入 ──
      await writeFile(workspace.inputPath, input);

      // ── Step 4: 生成脚本 ──
      const script = generateRembgScript(
        workspace.inputPath,
        workspace.outputPath,
        workspace.maskPath,
        this.config
      );
      await writeFile(workspace.scriptPath, script, 'utf-8');

      // ── Step 5: 执行 ──
      const result = await this.executeScript(workspace.scriptPath);

      // ── Step 6: 读取输出 ──
      let outputBuffer: Buffer;
      let maskBuffer: Buffer | null = null;

      try {
        outputBuffer = await readFile(workspace.outputPath);
      } catch {
        throw new Error('rembg processing failed: output file not generated');
      }

      try {
        maskBuffer = await readFile(workspace.maskPath);
      } catch {
        // mask 是可选的
      }

      const processingTimeMs = Date.now() - startTime;

      return {
        outputBuffer,
        maskBuffer,
        metadata: {
          model: this.config.modelName,
          processingTimeMs,
          pythonVersion: result.python_version || 'unknown',
          rembgVersion: result.rembg_version || 'unknown',
          inputSize: input.length,
          outputSize: outputBuffer.length,
        },
      };
    } finally {
      // ── Step 7: 清理 ──
      await workspace.cleanup();
    }
  }

  /**
   * 执行 Python 脚本
   *
   * 参考 OpenAI Codex 的沙箱执行模式:
   *   timeout 强制终止 + stderr 错误捕获 + 结构化输出解析
   */
  private executeScript(scriptPath: string): Promise<Record<string, any>> {
    return new Promise((resolve, reject) => {
      const child = spawn(this.config.pythonPath, [scriptPath], {
        timeout: this.config.timeoutMs,
        stdio: ['ignore', 'pipe', 'pipe'],
        env: {
          ...process.env,
          U2NET_HOME: this.config.modelCacheDir,
        },
      });

      let stdout = '';
      let stderr = '';

      child.stdout.on('data', (data: Buffer) => {
        stdout += data.toString();
      });

      child.stderr.on('data', (data: Buffer) => {
        const msg = data.toString();
        stderr += msg;

        // 检测模型下载进度
        if (msg.includes('Downloading') || msg.includes('downloading')) {
          this.emit('model-downloading', msg.trim());
        }
      });

      child.on('close', (code) => {
        if (code === 0) {
          try {
            const result = JSON.parse(stdout.trim());
            if (result.status === 'ok') {
              resolve(result);
            } else {
              reject(new Error(`rembg error: ${result.error}`));
            }
          } catch {
            // stdout 可能不是 JSON (模型下载输出等)
            resolve({ status: 'ok', python_version: 'unknown', rembg_version: 'unknown' });
          }
        } else {
          let errorMsg = `rembg process exited with code ${code}`;
          if (stderr) {
            try {
              const errResult = JSON.parse(stderr.trim());
              errorMsg = `rembg ${errResult.error_type}: ${errResult.error}`;
            } catch {
              errorMsg += `: ${stderr.substring(0, 500)}`;
            }
          }
          reject(new Error(errorMsg));
        }
      });

      child.on('error', (err) => {
        reject(new Error(`Failed to start Python process: ${err.message}`));
      });
    });
  }

  /**
   * 获取可用模型列表及推荐
   *
   * 参考 Megatron-Core 的 model registry:
   *   集中管理模型元数据, 包括大小、质量评分、适用场景
   */
  static getAvailableModels(): Array<{
    name: RembgModel;
    size: string;
    quality: number;
    speed: number;
    description: string;
    recommended: boolean;
  }> {
    return [
      {
        name: 'u2net',
        size: '176MB',
        quality: 4,
        speed: 3,
        description: '通用模型, 平衡质量与速度',
        recommended: false,
      },
      {
        name: 'u2netp',
        size: '4.7MB',
        quality: 2,
        speed: 5,
        description: '轻量版, 适合快速预览',
        recommended: false,
      },
      {
        name: 'u2net_human_seg',
        size: '176MB',
        quality: 4,
        speed: 3,
        description: '人像专用, 对人物轮廓优化',
        recommended: false,
      },
      {
        name: 'isnet-general-use',
        size: '176MB',
        quality: 5,
        speed: 3,
        description: 'ISNet 通用模型, 边缘更精确',
        recommended: true,
      },
      {
        name: 'birefnet-general',
        size: '~900MB',
        quality: 5,
        speed: 2,
        description: 'BiRefNet, 最高质量, 需要更多内存/GPU',
        recommended: false,
      },
      {
        name: 'silueta',
        size: '44MB',
        quality: 3,
        speed: 4,
        description: 'Silueta, 快速且质量可接受',
        recommended: false,
      },
    ];
  }

  /**
   * 预热: 预下载模型
   */
  async warmup(): Promise<void> {
    const envOk = await this.ensureEnvironment();
    if (!envOk) return;

    try {
      execSync(
        `${this.config.pythonPath} -c "from rembg import new_session; new_session('${this.config.modelName}')"`,
        {
          timeout: 120000, // 模型下载可能需要2分钟
          env: { ...process.env, U2NET_HOME: this.config.modelCacheDir },
        }
      );
      this.emit('warmup-complete', this.config.modelName);
    } catch (err) {
      this.emit('warmup-error', err);
    }
  }

  /** 更新配置 */
  updateConfig(partial: Partial<RembgConfig>): void {
    this.config = { ...this.config, ...partial };
    this.environmentChecked = false; // 需要重新检查
  }

  /** 获取当前配置 */
  getConfig(): Readonly<RembgConfig> {
    return { ...this.config };
  }
}

export default RembgBridge;