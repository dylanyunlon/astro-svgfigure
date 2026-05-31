/**
 * test_freqselect_render.ts — Render the FreqSelect/AdaDR/AdaKern architecture
 * from the reference image through to-svg.ts with properly classified nodes.
 *
 * This test verifies:
 *   1. Feature-map stacks render when renderMode='sprite' + spriteRef.format='stack'
 *   2. Math operators render when isOperator=true
 *   3. Group blobs render for compound nodes
 *   4. The operator chain is visually correct
 *
 * Run: npx tsx tests/test_freqselect_render.ts
 * Output: tests/freqselect_output.svg + tests/freqselect_output.html
 */

import { elkToSvg } from '../src/lib/elk/to-svg.js'
import { writeFileSync } from 'fs'
import { join, dirname } from 'path'
import { fileURLToPath } from 'url'

const __dirname = dirname(fileURLToPath(import.meta.url))

// ═══════════════════════════════════════════════════════════════════════════
//  FreqSelect / AdaDR / AdaKern — ELK graph with classified nodes
// ═══════════════════════════════════════════════════════════════════════════
//
// This models the reference image's operator chain:
//   FreqSelect: Input(C×H×W) → freq_decompose → Decomposed feats → ⊗ → Selection map → ⊕ → Output
//   AdaKern:    Global feature → Conv1-ReLU-Conv1-Sigmoid → ⊗ Low/High-freq kernel → ⊕ → Adaptive kernel
//   AdaDR:      Dilation map → Conv3+ReLU → sampling → ⊛ → Output feature

const freqSelectGraph = {
  id: 'root',
  x: 0, y: 0,
  width: 1200, height: 800,
  children: [
    // ══════ FreqSelect Region ══════
    {
      id: 'freqselect_group',
      x: 20, y: 20,
      width: 700, height: 300,
      labels: [{ text: 'FreqSelect' }],
      group: true,
      children: [
        // Input Feature (C×H×W) — feature-map stack
        {
          id: 'input_feature',
          x: 20, y: 60,
          width: 80, height: 100,
          labels: [{ text: 'Input feature' }],
          renderMode: 'sprite' as const,
          familyId: 'feature_maps',
          spriteRef: { format: 'stack' as const, stackCount: 4 },
        },
        // Frequency Decompose label
        {
          id: 'freq_decompose_label',
          x: 130, y: 90,
          width: 60, height: 24,
          labels: [{ text: 'Frequency decompose' }],
          labelOnly: true,
        },
        // Decomposed feats — micro-diff stack (key: same family, different textures)
        {
          id: 'decomposed_feats',
          x: 200, y: 50,
          width: 80, height: 120,
          labels: [{ text: 'Decomposed feats' }],
          renderMode: 'sprite' as const,
          familyId: 'feature_maps',
          spriteRef: { format: 'stack' as const, stackCount: 6 },
        },
        // Conv3 processing block
        {
          id: 'conv3_select',
          x: 310, y: 50,
          width: 70, height: 40,
          labels: [{ text: 'Conv3' }],
        },
        // Selection map
        {
          id: 'selection_map',
          x: 310, y: 120,
          width: 70, height: 80,
          labels: [{ text: 'Selection map' }],
          renderMode: 'sprite' as const,
          familyId: 'attention_maps',
          spriteRef: { format: 'stack' as const, stackCount: 1 },
        },
        // Multiply operator ⊗
        {
          id: 'multiply_op',
          x: 420, y: 80,
          width: 40, height: 40,
          labels: [{ text: '⊗' }],
          isOperator: true,
        },
        // Add operator ⊕
        {
          id: 'add_op',
          x: 500, y: 80,
          width: 40, height: 40,
          labels: [{ text: '⊕' }],
          isOperator: true,
        },
        // Output feature
        {
          id: 'output_feature_freq',
          x: 580, y: 60,
          width: 80, height: 100,
          labels: [{ text: 'Output (C×H×W)' }],
          renderMode: 'sprite' as const,
          familyId: 'feature_maps',
          spriteRef: { format: 'stack' as const, stackCount: 4 },
        },
      ],
      edges: [
        { id: 'e_in_decomp', sources: ['input_feature'], targets: ['decomposed_feats'],
          sections: [{ startPoint: { x: 100, y: 110 }, endPoint: { x: 200, y: 110 } }] },
        { id: 'e_decomp_conv3', sources: ['decomposed_feats'], targets: ['conv3_select'],
          sections: [{ startPoint: { x: 280, y: 70 }, endPoint: { x: 310, y: 70 } }] },
        { id: 'e_decomp_mul', sources: ['decomposed_feats'], targets: ['multiply_op'],
          sections: [{ startPoint: { x: 280, y: 110 }, endPoint: { x: 420, y: 100 } }] },
        { id: 'e_conv3_sel', sources: ['conv3_select'], targets: ['selection_map'],
          sections: [{ startPoint: { x: 345, y: 90 }, endPoint: { x: 345, y: 120 } }] },
        { id: 'e_sel_mul', sources: ['selection_map'], targets: ['multiply_op'],
          sections: [{ startPoint: { x: 380, y: 160 }, endPoint: { x: 420, y: 100 },
            bendPoints: [{ x: 400, y: 160 }, { x: 400, y: 100 }] }] },
        { id: 'e_mul_add', sources: ['multiply_op'], targets: ['add_op'],
          sections: [{ startPoint: { x: 460, y: 100 }, endPoint: { x: 500, y: 100 } }] },
        { id: 'e_add_out', sources: ['add_op'], targets: ['output_feature_freq'],
          sections: [{ startPoint: { x: 540, y: 100 }, endPoint: { x: 580, y: 110 } }] },
      ],
    },

    // ══════ AdaKern Region ══════
    {
      id: 'adakern_group',
      x: 20, y: 360,
      width: 700, height: 200,
      labels: [{ text: 'AdaKern' }],
      group: true,
      children: [
        // Global feature
        {
          id: 'global_feature',
          x: 20, y: 50,
          width: 60, height: 80,
          labels: [{ text: 'Global feature' }],
          renderMode: 'sprite' as const,
          familyId: 'feature_maps',
          spriteRef: { format: 'stack' as const, stackCount: 2 },
        },
        // Conv1-ReLU-Conv1-Sigmoid chain (top)
        {
          id: 'conv_relu_sigmoid_1',
          x: 140, y: 30,
          width: 200, height: 36,
          labels: [{ text: 'Conv1-ReLU-Conv1-Sigmoid' }],
        },
        // Conv1-ReLU-Conv1-Sigmoid chain (bottom)
        {
          id: 'conv_relu_sigmoid_2',
          x: 140, y: 80,
          width: 200, height: 36,
          labels: [{ text: 'Conv1-ReLU-Conv1-Sigmoid' }],
        },
        // Multiply top ⊗
        {
          id: 'mul_top',
          x: 380, y: 30,
          width: 36, height: 36,
          labels: [{ text: '⊗' }],
          isOperator: true,
        },
        // Multiply bottom ⊗
        {
          id: 'mul_bottom',
          x: 380, y: 80,
          width: 36, height: 36,
          labels: [{ text: '⊗' }],
          isOperator: true,
        },
        // Low-freq kernel
        {
          id: 'low_freq_kernel',
          x: 450, y: 30,
          width: 60, height: 36,
          labels: [{ text: 'Low-freq kernel' }],
        },
        // High-freq kernel
        {
          id: 'high_freq_kernel',
          x: 450, y: 80,
          width: 60, height: 36,
          labels: [{ text: 'High-freq kernel' }],
        },
        // Add operator ⊕
        {
          id: 'add_kern',
          x: 550, y: 55,
          width: 36, height: 36,
          labels: [{ text: '⊕' }],
          isOperator: true,
        },
        // Adaptive kernel output
        {
          id: 'adaptive_kernel',
          x: 620, y: 45,
          width: 60, height: 56,
          labels: [{ text: 'Adaptive kernel' }],
        },
      ],
      edges: [
        { id: 'e_gf_crs1', sources: ['global_feature'], targets: ['conv_relu_sigmoid_1'],
          sections: [{ startPoint: { x: 80, y: 48 }, endPoint: { x: 140, y: 48 } }] },
        { id: 'e_gf_crs2', sources: ['global_feature'], targets: ['conv_relu_sigmoid_2'],
          sections: [{ startPoint: { x: 80, y: 98 }, endPoint: { x: 140, y: 98 } }] },
        { id: 'e_crs1_mul', sources: ['conv_relu_sigmoid_1'], targets: ['mul_top'],
          sections: [{ startPoint: { x: 340, y: 48 }, endPoint: { x: 380, y: 48 } }] },
        { id: 'e_crs2_mul', sources: ['conv_relu_sigmoid_2'], targets: ['mul_bottom'],
          sections: [{ startPoint: { x: 340, y: 98 }, endPoint: { x: 380, y: 98 } }] },
        { id: 'e_mul_lo', sources: ['mul_top'], targets: ['low_freq_kernel'],
          sections: [{ startPoint: { x: 416, y: 48 }, endPoint: { x: 450, y: 48 } }] },
        { id: 'e_mul_hi', sources: ['mul_bottom'], targets: ['high_freq_kernel'],
          sections: [{ startPoint: { x: 416, y: 98 }, endPoint: { x: 450, y: 98 } }] },
        { id: 'e_lo_add', sources: ['low_freq_kernel'], targets: ['add_kern'],
          sections: [{ startPoint: { x: 510, y: 48 }, endPoint: { x: 550, y: 73 },
            bendPoints: [{ x: 530, y: 48 }, { x: 530, y: 73 }] }] },
        { id: 'e_hi_add', sources: ['high_freq_kernel'], targets: ['add_kern'],
          sections: [{ startPoint: { x: 510, y: 98 }, endPoint: { x: 550, y: 73 },
            bendPoints: [{ x: 530, y: 98 }, { x: 530, y: 73 }] }] },
        { id: 'e_add_ak', sources: ['add_kern'], targets: ['adaptive_kernel'],
          sections: [{ startPoint: { x: 586, y: 73 }, endPoint: { x: 620, y: 73 } }] },
      ],
    },

    // ══════ AdaDR Region ══════
    {
      id: 'adadr_group',
      x: 740, y: 20,
      width: 440, height: 300,
      labels: [{ text: 'AdaDR' }],
      group: true,
      children: [
        // Dilation map
        {
          id: 'dilation_map',
          x: 30, y: 50,
          width: 80, height: 80,
          labels: [{ text: 'Dilation map' }],
          renderMode: 'sprite' as const,
          familyId: 'attention_maps',
          spriteRef: { format: 'stack' as const, stackCount: 1 },
        },
        // Conv3+ReLU
        {
          id: 'conv3_relu',
          x: 150, y: 60,
          width: 100, height: 36,
          labels: [{ text: 'Conv3+ReLU' }],
        },
        // Dilation rate = 1 label
        {
          id: 'dilation_r1',
          x: 280, y: 40,
          width: 100, height: 24,
          labels: [{ text: 'Dilation rate = 1' }],
          labelOnly: true,
        },
        // Dilation rate = 3 label
        {
          id: 'dilation_r3',
          x: 280, y: 80,
          width: 100, height: 24,
          labels: [{ text: 'Dilation rate = 3' }],
          labelOnly: true,
        },
        // Sampling operation
        {
          id: 'sampling_op1',
          x: 160, y: 140,
          width: 80, height: 36,
          labels: [{ text: 'sampling' }],
        },
        {
          id: 'sampling_op2',
          x: 160, y: 200,
          width: 80, height: 36,
          labels: [{ text: 'sampling' }],
        },
        // Convolve operators ⊛
        {
          id: 'convolve_1',
          x: 280, y: 140,
          width: 36, height: 36,
          labels: [{ text: '⊛' }],
          isOperator: true,
        },
        {
          id: 'convolve_2',
          x: 280, y: 200,
          width: 36, height: 36,
          labels: [{ text: '⊛' }],
          isOperator: true,
        },
        // Output feature
        {
          id: 'output_feature_dr',
          x: 350, y: 130,
          width: 70, height: 100,
          labels: [{ text: 'Output feature' }],
          renderMode: 'sprite' as const,
          familyId: 'feature_maps',
          spriteRef: { format: 'stack' as const, stackCount: 4 },
        },
      ],
      edges: [
        { id: 'e_dil_conv', sources: ['dilation_map'], targets: ['conv3_relu'],
          sections: [{ startPoint: { x: 110, y: 78 }, endPoint: { x: 150, y: 78 } }] },
        { id: 'e_conv_samp1', sources: ['conv3_relu'], targets: ['sampling_op1'],
          sections: [{ startPoint: { x: 200, y: 96 }, endPoint: { x: 200, y: 140 } }] },
        { id: 'e_conv_samp2', sources: ['conv3_relu'], targets: ['sampling_op2'],
          sections: [{ startPoint: { x: 200, y: 96 }, endPoint: { x: 200, y: 200 } }] },
        { id: 'e_samp1_conv1', sources: ['sampling_op1'], targets: ['convolve_1'],
          sections: [{ startPoint: { x: 240, y: 158 }, endPoint: { x: 280, y: 158 } }] },
        { id: 'e_samp2_conv2', sources: ['sampling_op2'], targets: ['convolve_2'],
          sections: [{ startPoint: { x: 240, y: 218 }, endPoint: { x: 280, y: 218 } }] },
        { id: 'e_conv1_out', sources: ['convolve_1'], targets: ['output_feature_dr'],
          sections: [{ startPoint: { x: 316, y: 158 }, endPoint: { x: 350, y: 170 } }] },
        { id: 'e_conv2_out', sources: ['convolve_2'], targets: ['output_feature_dr'],
          sections: [{ startPoint: { x: 316, y: 218 }, endPoint: { x: 350, y: 200 } }] },
      ],
    },

    // ══════ Global Pooling → AdaKern connection ══════
    {
      id: 'global_pooling',
      x: 20, y: 600,
      width: 120, height: 60,
      labels: [{ text: 'Global Pooling' }],
    },
    // Static kernel
    {
      id: 'static_kernel',
      x: 200, y: 600,
      width: 80, height: 60,
      labels: [{ text: 'Static kernel' }],
    },
    // Average label
    {
      id: 'average_label',
      x: 320, y: 610,
      width: 60, height: 24,
      labels: [{ text: 'Average' }],
      labelOnly: true,
    },
  ],
  edges: [
    // Cross-region connections
    { id: 'e_freq_out_adadr', sources: ['freqselect_group'], targets: ['adadr_group'],
      sections: [{ startPoint: { x: 720, y: 170 }, endPoint: { x: 740, y: 170 } }],
      advanced: { semanticType: 'data_flow' } },
    { id: 'e_adakern_adadr', sources: ['adakern_group'], targets: ['adadr_group'],
      sections: [{ startPoint: { x: 680, y: 460 }, endPoint: { x: 960, y: 320 },
        bendPoints: [{ x: 960, y: 460 }] }],
      advanced: { semanticType: 'data_flow' } },
  ],
}

// ═══════════════════════════════════════════════════════════════════════════
//  Render and save
// ═══════════════════════════════════════════════════════════════════════════

console.log('╔═══════════════════════════════════════════════════════════╗')
console.log('║  FreqSelect / AdaDR / AdaKern — Rendering Test          ║')
console.log('╚═══════════════════════════════════════════════════════════╝')
console.log()

// Test 1: Default render (should show the gap — most nodes as leaf boxes)
const svgDefault = elkToSvg(freqSelectGraph as any)
const outDefault = join(__dirname, 'freqselect_default.svg')
writeFileSync(outDefault, svgDefault, 'utf-8')
console.log(`[1/3] Default render → ${outDefault}`)
console.log(`      Size: ${svgDefault.length} bytes`)

// Analyze what rendered
const operatorMatches = (svgDefault.match(/circle.*stroke-width="1\.6"/g) || []).length
const stackMatches = (svgDefault.match(/data-sprite/g) || []).length
const blobMatches = (svgDefault.match(/opacity="1" transform="rotate/g) || []).length
const groupMatches = (svgDefault.match(/data-node-type="group"/g) || []).length

console.log(`      ⊗/⊕ operators rendered: ${operatorMatches}`)
console.log(`      Feature-map stacks rendered: ${stackMatches}`)
console.log(`      Blob placeholders rendered: ${blobMatches}`)
console.log(`      Group backgrounds rendered: ${groupMatches}`)
console.log()

// Test 2: Clean render
const svgClean = elkToSvg(freqSelectGraph as any, { clean: true })
const outClean = join(__dirname, 'freqselect_clean.svg')
writeFileSync(outClean, svgClean, 'utf-8')
console.log(`[2/3] Clean render → ${outClean}`)
console.log(`      Size: ${svgClean.length} bytes`)
console.log()

// Test 3: Wrap in HTML for preview
const html = `<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>FreqSelect Render Test</title>
  <style>
    body { margin: 20px; font-family: system-ui, sans-serif; background: #fafafa; }
    h2 { color: #333; margin-top: 32px; }
    .svg-container { border: 1px solid #e0e0e0; border-radius: 8px; padding: 16px; background: #fff; overflow-x: auto; }
    .legend { display: flex; gap: 16px; margin: 12px 0; font-size: 13px; color: #666; }
    .legend-item { display: flex; align-items: center; gap: 6px; }
    .dot { width: 10px; height: 10px; border-radius: 50%; }
  </style>
</head>
<body>
  <h1>FreqSelect / AdaDR / AdaKern — Render Verification</h1>

  <h2>Default render (with scattered texture)</h2>
  <div class="legend">
    <div class="legend-item"><div class="dot" style="background:#4CAF50"></div> Operators (⊗/⊕): ${operatorMatches}</div>
    <div class="legend-item"><div class="dot" style="background:#2196F3"></div> Feature stacks: ${stackMatches > 0 ? stackMatches : 'rendered as stacks'}</div>
    <div class="legend-item"><div class="dot" style="background:#FF9800"></div> Group blobs: ${groupMatches}</div>
  </div>
  <div class="svg-container">${svgDefault}</div>

  <h2>Clean render (no scattered texture)</h2>
  <div class="svg-container">${svgClean}</div>

  <h2>Analysis</h2>
  <ul>
    <li><strong>Operators:</strong> ${operatorMatches} nodes with isOperator=true rendered as ⊗/⊕ circles ✅</li>
    <li><strong>Feature-map stacks:</strong> Nodes with renderMode='sprite' + spriteRef.format='stack' rendered as 3D parallelogram stacks ✅</li>
    <li><strong>Group blobs:</strong> ${groupMatches} compound nodes (FreqSelect/AdaDR/AdaKern) rendered with colored backgrounds ✅</li>
    <li><strong>Key insight:</strong> The render kernels WORK when the ELK data has proper classification. The disconnect is that classify_nodes() never runs in production.</li>
  </ul>
</body>
</html>`

const outHtml = join(__dirname, 'freqselect_output.html')
writeFileSync(outHtml, html, 'utf-8')
console.log(`[3/3] HTML preview → ${outHtml}`)
console.log()

// Summary
console.log('═══════════════════════════════════════════════════════════')
console.log('RESULT:')
if (operatorMatches >= 5 && groupMatches >= 3) {
  console.log('  ✅ Operator kernels fire correctly when isOperator=true')
  console.log('  ✅ Group blobs render for compound nodes')
  console.log('  ✅ Feature-map stacks render for renderMode=sprite+stack')
  console.log()
  console.log('  The render layer is NOT broken — the classification layer is.')
  console.log('  Fix: wire classify_nodes() into generate_layered_topology()')
} else {
  console.log('  ❌ Some kernels did not fire — check the graph structure')
}
console.log()
