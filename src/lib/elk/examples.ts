/**
 * examples.ts - Built-in topologies with advanced edge routing
 */

export const TRANSFORMER_EXAMPLE = {
  id: 'root',
  layoutOptions: { 'elk.algorithm': 'layered', 'elk.direction': 'DOWN', 'elk.spacing.nodeNode': '50', 'elk.edgeRouting': 'ORTHOGONAL' },
  children: [
    { id: 'input_embed', width: 160, height: 50, labels: [{ text: 'Input Embedding' }] },
    { id: 'pos_encode', width: 160, height: 50, labels: [{ text: 'Positional Encoding' }] },
    { id: 'self_attn', width: 160, height: 50, labels: [{ text: 'Multi-Head Attention' }] },
    { id: 'add_norm1', width: 140, height: 40, labels: [{ text: 'Add and Norm' }] },
    { id: 'ffn', width: 140, height: 45, labels: [{ text: 'Feed Forward' }] },
    { id: 'add_norm2', width: 140, height: 40, labels: [{ text: 'Add and Norm' }] },
    { id: 'output', width: 160, height: 50, labels: [{ text: 'Output' }] },
  ],
  edges: [
    { id: 'e1', sources: ['input_embed'], targets: ['pos_encode'] },
    { id: 'e2', sources: ['pos_encode'], targets: ['self_attn'] },
    { id: 'e3', sources: ['self_attn'], targets: ['add_norm1'] },
    { id: 'e4', sources: ['add_norm1'], targets: ['ffn'] },
    { id: 'e5', sources: ['ffn'], targets: ['add_norm2'] },
    { id: 'e6', sources: ['add_norm2'], targets: ['output'] },
    { id: 'skip1', sources: ['pos_encode'], targets: ['add_norm1'],
      advanced: { semanticType: 'skip_connection', routing: 'SPLINES', curvature: 0.6, strokeColor: '#4CAF50', strokeWidth: 2 } },
    { id: 'skip2', sources: ['add_norm1'], targets: ['add_norm2'],
      advanced: { semanticType: 'skip_connection', routing: 'SPLINES', curvature: 0.6, strokeColor: '#4CAF50', strokeWidth: 2 } },
  ],
}

export const VAE_EXAMPLE = {
  id: 'root',
  layoutOptions: { 'elk.algorithm': 'layered', 'elk.direction': 'RIGHT', 'elk.spacing.nodeNode': '60' },
  children: [
    { id: 'input', width: 120, height: 50, labels: [{ text: 'Input x' }] },
    { id: 'encoder', width: 150, height: 60, labels: [{ text: 'Encoder' }] },
    { id: 'mu', width: 100, height: 45, labels: [{ text: 'mu' }] },
    { id: 'sigma', width: 100, height: 45, labels: [{ text: 'sigma' }] },
    { id: 'sample', width: 120, height: 50, labels: [{ text: 'Reparameterize' }] },
    { id: 'decoder', width: 150, height: 60, labels: [{ text: 'Decoder' }] },
    { id: 'output', width: 120, height: 50, labels: [{ text: 'x_hat' }] },
  ],
  edges: [
    { id: 'e1', sources: ['input'], targets: ['encoder'] },
    { id: 'e2', sources: ['encoder'], targets: ['mu'], advanced: { semanticType: 'fan_out' } },
    { id: 'e3', sources: ['encoder'], targets: ['sigma'], advanced: { semanticType: 'fan_out' } },
    { id: 'e4', sources: ['mu'], targets: ['sample'], advanced: { semanticType: 'fan_in' } },
    { id: 'e5', sources: ['sigma'], targets: ['sample'], advanced: { semanticType: 'fan_in' } },
    { id: 'e6', sources: ['sample'], targets: ['decoder'],
      advanced: { semanticType: 'data_flow', edgeLabels: [{ text: 'z~N(mu,sigma)', position: 0.5 }] } },
    { id: 'e7', sources: ['decoder'], targets: ['output'] },
  ],
}

export const RESNET_EXAMPLE = {
  id: 'root',
  layoutOptions: { 'elk.algorithm': 'layered', 'elk.direction': 'DOWN' },
  children: [
    { id: 'input', width: 140, height: 45, labels: [{ text: 'Input' }] },
    { id: 'conv1', width: 140, height: 45, labels: [{ text: 'Conv 3x3' }] },
    { id: 'bn1', width: 140, height: 40, labels: [{ text: 'BN+ReLU' }] },
    { id: 'conv2', width: 140, height: 45, labels: [{ text: 'Conv 3x3' }] },
    { id: 'bn2', width: 140, height: 40, labels: [{ text: 'BatchNorm' }] },
    { id: 'add', width: 100, height: 50, labels: [{ text: '+' }] },
    { id: 'relu', width: 140, height: 40, labels: [{ text: 'ReLU' }] },
  ],
  edges: [
    { id: 'e1', sources: ['input'], targets: ['conv1'] },
    { id: 'e2', sources: ['conv1'], targets: ['bn1'] },
    { id: 'e3', sources: ['bn1'], targets: ['conv2'] },
    { id: 'e4', sources: ['conv2'], targets: ['bn2'] },
    { id: 'e5', sources: ['bn2'], targets: ['add'] },
    { id: 'skip', sources: ['input'], targets: ['add'],
      advanced: { semanticType: 'skip_connection', routing: 'SPLINES', curvature: 0.8,
        strokeColor: '#4CAF50', strokeWidth: 2.5, edgeLabels: [{ text: 'identity', position: 0.5 }] } },
    { id: 'e6', sources: ['add'], targets: ['relu'] },
  ],
}

export const EXAMPLES = {
  transformer: { name: 'Transformer', graph: TRANSFORMER_EXAMPLE },
  vae: { name: 'VAE', graph: VAE_EXAMPLE },
  resnet: { name: 'ResNet', graph: RESNET_EXAMPLE },
}
export default EXAMPLES
