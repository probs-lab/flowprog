# Vue.js Migration Plan for FlowProg Visualization

This document provides a detailed step-by-step plan to migrate the FlowProg visualization app from vanilla JavaScript to Vue.js.

## Overview

**Current State:**
- 557-line monolithic `app.js`
- 340-line HTML template with inline CSS
- Manual DOM manipulation with string building
- Global state management

**Target State:**
- Modular Vue components (~50-100 lines each)
- Component-scoped styling
- Reactive state management
- Reusable, testable components

**Estimated Time:** 6-8 hours for full migration
**Skill Level Required:** Intermediate JavaScript, basic Vue knowledge

---

## Phase 1: Project Setup (1 hour)

### Step 1.1: Install Node.js and npm

Ensure you have Node.js 18+ installed:
```bash
node --version  # Should be v18.0.0 or higher
npm --version   # Should be v9.0.0 or higher
```

### Step 1.2: Create Vue Project with Vite

In the `src/flowprog/visualization/` directory:

```bash
cd src/flowprog/visualization/
npm create vite@latest frontend -- --template vue
cd frontend
npm install
```

### Step 1.3: Install Dependencies

```bash
# Core dependencies
npm install cytoscape cytoscape-dagre dagre

# Optional but recommended
npm install axios  # For cleaner API calls
```

### Step 1.4: Configure Vite for Flask Integration

Edit `frontend/vite.config.js`:

```javascript
import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'

export default defineConfig({
  plugins: [vue()],
  server: {
    port: 3000,
    proxy: {
      // Proxy API calls to Flask backend during development
      '/api': {
        target: 'http://localhost:5000',
        changeOrigin: true
      }
    }
  },
  build: {
    // Output to Flask's static directory
    outDir: '../static/dist',
    emptyOutDir: true,
    rollupOptions: {
      output: {
        // Keep consistent filenames for Flask template
        entryFileNames: 'assets/[name].js',
        chunkFileNames: 'assets/[name].js',
        assetFileNames: 'assets/[name].[ext]'
      }
    }
  }
})
```

### Step 1.5: Update Flask Template

Create new `templates/index-vue.html`:

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>FlowProg Model Visualization</title>

    <!-- MathJax for LaTeX rendering -->
    <script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>

    <!-- Development: Vite dev server -->
    {% if config.ENV == 'development' %}
    <script type="module" src="http://localhost:3000/@vite/client"></script>
    <script type="module" src="http://localhost:3000/src/main.js"></script>
    {% else %}
    <!-- Production: Built assets -->
    <script type="module" src="{{ url_for('static', filename='dist/assets/main.js') }}"></script>
    <link rel="stylesheet" href="{{ url_for('static', filename='dist/assets/main.css') }}">
    {% endif %}
</head>
<body>
    <div id="app"></div>
</body>
</html>
```

### Step 1.6: Update Flask Server

Edit `server.py` to add development mode support:

```python
def __init__(self, model, recipe_data: Dict = None, parameter_values: Dict = None, dev_mode: bool = False):
    # ... existing code ...
    self.app = Flask(__name__,
                    template_folder='templates',
                    static_folder='static')
    self.app.config['ENV'] = 'development' if dev_mode else 'production'
    CORS(self.app)

    # Register routes
    self._register_routes()

def _register_routes(self):
    @self.app.route('/')
    def index():
        return render_template('index-vue.html')

    # ... rest of routes unchanged ...
```

---

## Phase 2: Create Core Architecture (1 hour)

### Step 2.1: Project Structure

Create this directory structure in `frontend/src/`:

```
src/
├── components/
│   ├── graph/
│   │   ├── GraphView.vue
│   │   └── ControlBar.vue
│   ├── panels/
│   │   ├── DetailsPanel.vue
│   │   ├── ProcessPanel.vue
│   │   └── FlowPanel.vue
│   └── expressions/
│       ├── ExpressionBox.vue
│       ├── ExpressionModes.vue
│       ├── IntermediatesList.vue
│       └── HistoryList.vue
├── composables/
│   ├── useGraphData.js
│   └── useMathJax.js
├── stores/
│   └── graphStore.js
├── App.vue
├── main.js
└── style.css
```

### Step 2.2: Create Main Entry Point

`frontend/src/main.js`:

```javascript
import { createApp } from 'vue'
import App from './App.vue'
import './style.css'

const app = createApp(App)
app.mount('#app')
```

### Step 2.3: Create Global Styles

`frontend/src/style.css`:

```css
* {
  margin: 0;
  padding: 0;
  box-sizing: border-box;
}

body {
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
  height: 100vh;
  overflow: hidden;
  background: #f5f5f5;
}

#app {
  height: 100vh;
  display: flex;
  flex-direction: column;
}
```

### Step 2.4: Create State Management Store

`frontend/src/stores/graphStore.js`:

```javascript
import { reactive, readonly } from 'vue'

const state = reactive({
  graphData: null,
  selectedElement: null,
  panelData: null,
  isPanelOpen: false,
  isLoading: false,
  error: null
})

const actions = {
  setGraphData(data) {
    state.graphData = data
  },

  selectElement(element) {
    state.selectedElement = element
  },

  setPanelData(data) {
    state.panelData = data
    state.isPanelOpen = !!data
  },

  closePanel() {
    state.panelData = null
    state.isPanelOpen = false
    state.selectedElement = null
  },

  setLoading(loading) {
    state.isLoading = loading
  },

  setError(error) {
    state.error = error
  }
}

export const useGraphStore = () => {
  return {
    state: readonly(state),
    ...actions
  }
}
```

---

## Phase 3: Build Core Components (2-3 hours)

### Step 3.1: Main App Component

`frontend/src/App.vue`:

```vue
<template>
  <div class="app-container">
    <header>
      <h1>FlowProg Model Visualization</h1>
      <p>Interactive exploration of process flows and model expressions</p>
    </header>

    <main>
      <GraphView
        @process-clicked="handleProcessClick"
        @flow-clicked="handleFlowClick"
        @background-clicked="handleBackgroundClick"
      />

      <DetailsPanel
        v-if="store.state.isPanelOpen"
        :data="store.state.panelData"
        @close="store.closePanel"
      />
    </main>
  </div>
</template>

<script setup>
import { onMounted } from 'vue'
import GraphView from './components/graph/GraphView.vue'
import DetailsPanel from './components/panels/DetailsPanel.vue'
import { useGraphStore } from './stores/graphStore'
import { useGraphData } from './composables/useGraphData'

const store = useGraphStore()
const { loadGraphData, loadProcessDetails, loadFlowDetails } = useGraphData()

onMounted(async () => {
  await loadGraphData()
})

const handleProcessClick = async (processId) => {
  const data = await loadProcessDetails(processId)
  if (data && !data.error) {
    store.setPanelData({ type: 'process', ...data })
  }
}

const handleFlowClick = async ({ source, target, material }) => {
  const data = await loadFlowDetails(source, target, material)
  if (data && !data.error) {
    store.setPanelData({ type: 'flow', ...data })
  }
}

const handleBackgroundClick = () => {
  store.closePanel()
}
</script>

<style scoped>
.app-container {
  display: flex;
  flex-direction: column;
  height: 100vh;
}

header {
  background: #2c3e50;
  color: white;
  padding: 1rem 2rem;
  box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

header h1 {
  font-size: 1.5rem;
  font-weight: 600;
}

header p {
  font-size: 0.9rem;
  opacity: 0.8;
  margin-top: 0.25rem;
}

main {
  position: relative;
  flex: 1;
  overflow: hidden;
}
</style>
```

### Step 3.2: Graph View Component

`frontend/src/components/graph/GraphView.vue`:

```vue
<template>
  <div class="graph-container">
    <div ref="cyContainer" class="cy-container"></div>
    <ControlBar @fit="fitGraph" @reset="resetLayout" />
  </div>
</template>

<script setup>
import { ref, onMounted, watch } from 'vue'
import cytoscape from 'cytoscape'
import dagre from 'dagre'
import cytoscapeDagre from 'cytoscape-dagre'
import { useGraphStore } from '../../stores/graphStore'
import ControlBar from './ControlBar.vue'

// Register dagre layout
cytoscape.use(cytoscapeDagre)

const emit = defineEmits(['process-clicked', 'flow-clicked', 'background-clicked'])
const store = useGraphStore()

const cyContainer = ref(null)
let cy = null

const cytoscapeStyles = [
  {
    selector: 'node.process',
    style: {
      'background-color': '#3498db',
      'label': 'data(label)',
      'color': '#fff',
      'text-valign': 'center',
      'text-halign': 'center',
      'font-size': '12px',
      'font-weight': 'bold',
      'width': '80px',
      'height': '60px',
      'shape': 'rectangle',
      'border-width': 2,
      'border-color': '#2980b9',
      'text-wrap': 'wrap',
      'text-max-width': '75px'
    }
  },
  {
    selector: 'node.object',
    style: {
      'background-color': '#95a5a6',
      'label': 'data(label)',
      'color': '#2c3e50',
      'text-valign': 'center',
      'text-halign': 'center',
      'font-size': '11px',
      'width': '60px',
      'height': '60px',
      'shape': 'ellipse',
      'border-width': 2,
      'border-color': '#7f8c8d',
      'text-wrap': 'wrap',
      'text-max-width': '55px'
    }
  },
  {
    selector: 'node.market',
    style: {
      'background-color': '#e74c3c',
      'border-color': '#c0392b',
      'color': '#fff'
    }
  },
  {
    selector: 'edge.flow',
    style: {
      'width': 3,
      'line-color': '#34495e',
      'target-arrow-color': '#34495e',
      'target-arrow-shape': 'triangle',
      'curve-style': 'bezier',
      'label': 'data(label)',
      'font-size': '10px',
      'text-rotation': 'autorotate',
      'text-margin-y': -10,
      'color': '#2c3e50',
      'text-background-color': '#fff',
      'text-background-opacity': 0.8,
      'text-background-padding': '3px',
      'text-wrap': 'wrap'
    }
  },
  {
    selector: ':selected',
    style: {
      'border-width': 4,
      'border-color': '#f39c12',
      'line-color': '#f39c12',
      'target-arrow-color': '#f39c12',
      'z-index': 999
    }
  },
  {
    selector: 'node:active',
    style: {
      'overlay-color': '#f39c12',
      'overlay-padding': 8,
      'overlay-opacity': 0.3
    }
  },
  {
    selector: 'edge:active',
    style: {
      'overlay-color': '#f39c12',
      'overlay-padding': 4,
      'overlay-opacity': 0.3
    }
  }
]

const initializeCytoscape = () => {
  cy = cytoscape({
    container: cyContainer.value,
    style: cytoscapeStyles,
    layout: {
      name: 'dagre',
      rankDir: 'LR',
      nodeSep: 50,
      rankSep: 100,
      padding: 30
    }
  })

  // Event handlers
  cy.on('tap', 'node.process', (evt) => {
    const node = evt.target
    emit('process-clicked', node.data('id'))
  })

  cy.on('tap', 'edge.flow', (evt) => {
    const edge = evt.target
    emit('flow-clicked', {
      source: edge.data('source'),
      target: edge.data('target'),
      material: edge.data('material')
    })
  })

  cy.on('tap', (evt) => {
    if (evt.target === cy) {
      emit('background-clicked')
    }
  })
}

const updateGraph = () => {
  if (!cy || !store.state.graphData) return

  cy.elements().remove()
  cy.add(store.state.graphData.nodes)
  cy.add(store.state.graphData.edges)

  cy.layout({
    name: 'dagre',
    rankDir: 'LR',
    nodeSep: 50,
    rankSep: 100,
    padding: 30
  }).run()

  cy.fit(null, 50)
}

const fitGraph = () => {
  if (cy) cy.fit(null, 50)
}

const resetLayout = () => {
  if (cy) {
    cy.layout({
      name: 'dagre',
      rankDir: 'LR',
      nodeSep: 50,
      rankSep: 100,
      padding: 30
    }).run()
  }
}

onMounted(() => {
  initializeCytoscape()
  updateGraph()
})

watch(() => store.state.graphData, () => {
  updateGraph()
})

// Expose methods for parent components if needed
defineExpose({ fitGraph, resetLayout })
</script>

<style scoped>
.graph-container {
  width: 100%;
  height: 100%;
  position: relative;
  background: white;
}

.cy-container {
  width: 100%;
  height: 100%;
}
</style>
```

### Step 3.3: Control Bar Component

`frontend/src/components/graph/ControlBar.vue`:

```vue
<template>
  <div class="controls">
    <button @click="$emit('fit')">Fit to Screen</button>
    <button @click="$emit('reset')">Reset Layout</button>
  </div>
</template>

<script setup>
defineEmits(['fit', 'reset'])
</script>

<style scoped>
.controls {
  position: absolute;
  top: 1rem;
  right: 1rem;
  background: white;
  padding: 0.75rem;
  border-radius: 8px;
  box-shadow: 0 2px 8px rgba(0,0,0,0.15);
  z-index: 1000;
}

button {
  background: #3498db;
  color: white;
  border: none;
  padding: 0.5rem 1rem;
  border-radius: 4px;
  cursor: pointer;
  font-size: 0.9rem;
  margin: 0.25rem;
}

button:hover {
  background: #2980b9;
}
</style>
```

### Step 3.4: Details Panel Container

`frontend/src/components/panels/DetailsPanel.vue`:

```vue
<template>
  <div :class="['details-panel', { 'collapsed': !isOpen }]">
    <div class="panel-header">
      <span class="panel-title">{{ title }}</span>
      <button class="close-btn" @click="$emit('close')">&times;</button>
    </div>

    <div class="panel-content">
      <ProcessPanel v-if="data.type === 'process'" :data="data" />
      <FlowPanel v-if="data.type === 'flow'" :data="data" />
    </div>
  </div>
</template>

<script setup>
import { computed } from 'vue'
import ProcessPanel from './ProcessPanel.vue'
import FlowPanel from './FlowPanel.vue'

const props = defineProps({
  data: {
    type: Object,
    required: true
  }
})

defineEmits(['close'])

const isOpen = computed(() => !!props.data)

const title = computed(() => {
  if (props.data.type === 'process') {
    return `Process: ${props.data.process_id}`
  } else if (props.data.type === 'flow') {
    return `Flow: ${props.data.material}`
  }
  return 'Details'
})
</script>

<style scoped>
.details-panel {
  position: absolute;
  top: 0;
  right: 0;
  width: 500px;
  height: 100%;
  background: white;
  overflow-y: auto;
  box-shadow: -2px 0 8px rgba(0,0,0,0.1);
  display: flex;
  flex-direction: column;
  transform: translateX(0);
  transition: transform 0.3s ease-in-out;
  z-index: 2000;
}

.details-panel.collapsed {
  transform: translateX(100%);
}

.panel-header {
  background: #34495e;
  color: white;
  padding: 1rem;
  font-size: 1.1rem;
  font-weight: 600;
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.close-btn {
  background: none;
  border: none;
  color: white;
  font-size: 1.5rem;
  cursor: pointer;
  padding: 0;
  width: 30px;
  height: 30px;
  display: flex;
  align-items: center;
  justify-content: center;
  border-radius: 4px;
}

.close-btn:hover {
  background: rgba(255,255,255,0.1);
}

.panel-content {
  padding: 1.5rem;
  flex: 1;
  overflow-y: auto;
}
</style>
```

### Step 3.5: Process Panel Component

`frontend/src/components/panels/ProcessPanel.vue`:

```vue
<template>
  <div class="process-panel">
    <!-- Process Information -->
    <section class="section">
      <h3 class="section-title">Process Information</h3>
      <div class="info-grid">
        <div class="info-label">ID:</div>
        <div class="info-value">{{ data.process_id }}</div>
        <div class="info-label">Has Stock:</div>
        <div class="info-value">{{ data.has_stock ? 'Yes' : 'No' }}</div>
        <div class="info-label">Consumes:</div>
        <div class="info-value">{{ data.consumes.join(', ') }}</div>
        <div class="info-label">Produces:</div>
        <div class="info-value">{{ data.produces.join(', ') }}</div>
      </div>
    </section>

    <!-- Input Flows -->
    <section v-if="data.inputs.length > 0" class="section">
      <h3 class="section-title">Input Flows</h3>
      <div v-for="input in data.inputs" :key="input.object" class="flow-item">
        <div class="flow-object">{{ input.object }}</div>
        <div class="flow-value">Value: {{ input.value }}</div>
      </div>
    </section>

    <!-- Output Flows -->
    <section v-if="data.outputs.length > 0" class="section">
      <h3 class="section-title">Output Flows</h3>
      <div v-for="output in data.outputs" :key="output.object" class="flow-item">
        <div class="flow-object">{{ output.object }}</div>
        <div class="flow-value">Value: {{ output.value }}</div>
      </div>
    </section>

    <!-- X and Y Expressions -->
    <template v-if="!data.has_stock && xEqualsY">
      <section class="section">
        <h3 class="section-title">Process Activity (X = Y, no stock)</h3>
        <p class="note">
          This process has no stock, so input magnitude (X) equals output magnitude (Y).
        </p>
        <ExpressionModes :analysis="data.x_analysis" />
      </section>
    </template>

    <template v-else>
      <section class="section">
        <h3 class="section-title">X (Process Input Magnitude)</h3>
        <ExpressionModes :analysis="data.x_analysis" />
      </section>

      <section class="section">
        <h3 class="section-title">Y (Process Output Magnitude)</h3>
        <ExpressionModes :analysis="data.y_analysis" />
      </section>
    </template>
  </div>
</template>

<script setup>
import { computed, onMounted } from 'vue'
import ExpressionModes from '../expressions/ExpressionModes.vue'
import { useMathJax } from '../../composables/useMathJax'

const props = defineProps({
  data: {
    type: Object,
    required: true
  }
})

const { renderMath } = useMathJax()

const xEqualsY = computed(() => {
  return props.data.x_analysis.final_expression === props.data.y_analysis.final_expression
})

onMounted(() => {
  renderMath()
})
</script>

<style scoped>
.section {
  margin-bottom: 2rem;
}

.section-title {
  font-size: 1rem;
  font-weight: 600;
  color: #2c3e50;
  margin-bottom: 0.75rem;
  padding-bottom: 0.5rem;
  border-bottom: 2px solid #3498db;
}

.info-grid {
  display: grid;
  grid-template-columns: auto 1fr;
  gap: 0.5rem 1rem;
  margin: 0.75rem 0;
}

.info-label {
  font-weight: 600;
  color: #7f8c8d;
}

.info-value {
  color: #2c3e50;
}

.flow-item {
  background: #fff;
  border: 1px solid #e0e0e0;
  padding: 0.75rem;
  margin: 0.5rem 0;
  border-radius: 4px;
}

.flow-object {
  font-weight: 600;
  color: #2c3e50;
  margin-bottom: 0.25rem;
}

.flow-value {
  font-family: 'Courier New', monospace;
  font-size: 0.85rem;
  color: #27ae60;
}

.note {
  font-size: 0.85rem;
  color: #7f8c8d;
  margin-bottom: 0.75rem;
}
</style>
```

### Step 3.6: Flow Panel Component

`frontend/src/components/panels/FlowPanel.vue`:

```vue
<template>
  <div class="flow-panel">
    <!-- Flow Information -->
    <section class="section">
      <h3 class="section-title">Flow Information</h3>
      <div class="info-grid">
        <div class="info-label">Material:</div>
        <div class="info-value">{{ data.material }}</div>
        <div class="info-label">Type:</div>
        <div class="info-value">{{ data.flow_type }}</div>
        <div class="info-label">From:</div>
        <div class="info-value">{{ data.source }}</div>
        <div class="info-label">To:</div>
        <div class="info-value">{{ data.target }}</div>
        <div class="info-label">Value:</div>
        <div class="info-value">{{ data.evaluated_value }}</div>
      </div>
      <p class="description">{{ data.description }}</p>
    </section>

    <!-- Flow Expression -->
    <section class="section">
      <h3 class="section-title">Flow Expression</h3>
      <ExpressionModes :analysis="data.analysis" />
    </section>
  </div>
</template>

<script setup>
import { onMounted } from 'vue'
import ExpressionModes from '../expressions/ExpressionModes.vue'
import { useMathJax } from '../../composables/useMathJax'

defineProps({
  data: {
    type: Object,
    required: true
  }
})

const { renderMath } = useMathJax()

onMounted(() => {
  renderMath()
})
</script>

<style scoped>
.section {
  margin-bottom: 2rem;
}

.section-title {
  font-size: 1rem;
  font-weight: 600;
  color: #2c3e50;
  margin-bottom: 0.75rem;
  padding-bottom: 0.5rem;
  border-bottom: 2px solid #3498db;
}

.info-grid {
  display: grid;
  grid-template-columns: auto 1fr;
  gap: 0.5rem 1rem;
  margin: 0.75rem 0;
}

.info-label {
  font-weight: 600;
  color: #7f8c8d;
}

.info-value {
  color: #2c3e50;
}

.description {
  margin-top: 1rem;
  color: #7f8c8d;
  font-size: 0.9rem;
}
</style>
```

### Step 3.7: Expression Modes Component

`frontend/src/components/expressions/ExpressionModes.vue`:

```vue
<template>
  <div class="expression-modes">
    <!-- Fully Symbolic -->
    <ExpressionBox
      label="Expression (Fully Symbolic)"
      :expression="analysis.evaluation_modes?.symbolic || analysis.final_expression"
      :latex="analysis.evaluation_modes?.symbolic_latex || analysis.final_latex"
      note="No substitution - shows model structure"
    />

    <!-- Recipe Evaluated -->
    <ExpressionBox
      label="Expression (Recipe Evaluated)"
      :expression="analysis.evaluation_modes?.recipe_evaluated || analysis.final_expression"
      :latex="analysis.evaluation_modes?.recipe_latex || analysis.final_latex"
      note="Recipe coefficients (S, U) substituted"
      :warning="missingCoefficientsWarning"
    />

    <!-- Fully Evaluated -->
    <ExpressionBox
      label="Expression (Fully Evaluated)"
      :expression="analysis.evaluation_modes?.fully_evaluated || analysis.final_expression"
      :latex="analysis.evaluation_modes?.fully_latex || analysis.final_latex"
      note="All parameters and intermediates substituted"
    />

    <!-- History -->
    <HistoryList
      v-if="analysis.history && analysis.history.length > 0"
      :history="analysis.history"
    />

    <!-- Intermediates -->
    <IntermediatesList
      v-if="analysis.intermediates && analysis.intermediates.length > 0"
      :intermediates="analysis.intermediates"
    />
  </div>
</template>

<script setup>
import { computed } from 'vue'
import ExpressionBox from './ExpressionBox.vue'
import HistoryList from './HistoryList.vue'
import IntermediatesList from './IntermediatesList.vue'

const props = defineProps({
  analysis: {
    type: Object,
    required: true
  }
})

const missingCoefficientsWarning = computed(() => {
  const missing = props.analysis.evaluation_modes?.missing_coefficients
  if (!missing || missing.length === 0) return null

  return {
    title: 'Warning: Missing Recipe Coefficients',
    message: `The following coefficients are not defined in recipe_data: ${missing.join(', ')}`,
    detail: 'This typically indicates a typo in the recipe_data dictionary or missing coefficient definitions.'
  }
})
</script>
```

### Step 3.8: Expression Box Component (Reusable)

`frontend/src/components/expressions/ExpressionBox.vue`:

```vue
<template>
  <div class="expression-box">
    <div class="expression-label">{{ label }}</div>
    <div class="expression-content">{{ expression }}</div>
    <div class="latex-content" v-html="latexHtml"></div>
    <div v-if="note" class="note">{{ note }}</div>

    <!-- Warning Box -->
    <div v-if="warning" class="warning-box">
      <div class="warning-title">⚠️ {{ warning.title }}</div>
      <div class="warning-message">{{ warning.message }}</div>
      <div class="warning-detail">{{ warning.detail }}</div>
    </div>
  </div>
</template>

<script setup>
import { computed, onMounted, watch } from 'vue'
import { useMathJax } from '../../composables/useMathJax'

const props = defineProps({
  label: String,
  expression: String,
  latex: String,
  note: String,
  warning: Object
})

const { renderMath } = useMathJax()

const latexHtml = computed(() => `$$${props.latex}$$`)

// Re-render math when latex changes
watch(() => props.latex, () => {
  renderMath()
})

onMounted(() => {
  renderMath()
})
</script>

<style scoped>
.expression-box {
  background: #f8f9fa;
  border: 1px solid #e0e0e0;
  border-radius: 6px;
  padding: 1rem;
  margin: 0.75rem 0;
}

.expression-label {
  font-size: 0.85rem;
  color: #7f8c8d;
  margin-bottom: 0.5rem;
  font-weight: 500;
}

.expression-content {
  font-family: 'Courier New', monospace;
  font-size: 0.9rem;
  color: #2c3e50;
  overflow-x: auto;
  padding: 0.5rem;
  background: white;
  border-radius: 4px;
}

.latex-content {
  margin-top: 0.5rem;
  padding: 0.75rem;
  background: white;
  border-radius: 4px;
  overflow-x: auto;
}

.note {
  font-size: 0.85rem;
  color: #7f8c8d;
  margin-top: 0.5rem;
}

.warning-box {
  background: #fff3cd;
  border: 1px solid #ffc107;
  border-radius: 4px;
  padding: 0.75rem;
  margin-top: 0.5rem;
}

.warning-title {
  color: #856404;
  font-weight: 600;
  font-size: 0.9rem;
  margin-bottom: 0.25rem;
}

.warning-message {
  color: #856404;
  font-size: 0.85rem;
}

.warning-detail {
  color: #856404;
  font-size: 0.8rem;
  margin-top: 0.5rem;
  font-style: italic;
}
</style>
```

### Step 3.9: History List Component

`frontend/src/components/expressions/HistoryList.vue`:

```vue
<template>
  <div class="history-section">
    <div class="expression-label">Modeling History</div>
    <p class="note">Steps that contributed to this expression:</p>
    <div v-for="(step, index) in history" :key="index" class="history-item">
      {{ step }}
    </div>
  </div>
</template>

<script setup>
defineProps({
  history: {
    type: Array,
    required: true
  }
})
</script>

<style scoped>
.history-section {
  margin-top: 1.5rem;
}

.expression-label {
  font-size: 0.85rem;
  color: #7f8c8d;
  margin-bottom: 0.5rem;
  font-weight: 500;
}

.note {
  font-size: 0.85rem;
  color: #7f8c8d;
  margin-bottom: 0.5rem;
}

.history-item {
  background: #e8f5e9;
  padding: 0.75rem;
  margin: 0.5rem 0;
  border-radius: 4px;
  border-left: 3px solid #27ae60;
}
</style>
```

### Step 3.10: Intermediates List Component

`frontend/src/components/expressions/IntermediatesList.vue`:

```vue
<template>
  <div class="intermediates-section">
    <div class="expression-label">Intermediate Variables</div>
    <p class="note">Breakdown of intermediate calculations:</p>
    <div v-for="inter in intermediates" :key="inter.symbol" class="intermediate-item">
      <div class="intermediate-symbol">{{ inter.symbol }}</div>
      <div class="intermediate-description">{{ inter.description }}</div>
      <div class="intermediate-value">{{ inter.value }}</div>
      <div class="latex-content" v-html="`$$${inter.symbol} = ${inter.value_latex}$$`"></div>
    </div>
  </div>
</template>

<script setup>
import { onMounted, watch } from 'vue'
import { useMathJax } from '../../composables/useMathJax'

const props = defineProps({
  intermediates: {
    type: Array,
    required: true
  }
})

const { renderMath } = useMathJax()

watch(() => props.intermediates, () => {
  renderMath()
})

onMounted(() => {
  renderMath()
})
</script>

<style scoped>
.intermediates-section {
  margin-top: 1.5rem;
}

.expression-label {
  font-size: 0.85rem;
  color: #7f8c8d;
  margin-bottom: 0.5rem;
  font-weight: 500;
}

.note {
  font-size: 0.85rem;
  color: #7f8c8d;
  margin-bottom: 0.5rem;
}

.intermediate-item {
  background: #fff;
  border-left: 3px solid #3498db;
  padding: 0.75rem;
  margin: 0.5rem 0;
  border-radius: 4px;
}

.intermediate-symbol {
  font-family: 'Courier New', monospace;
  font-weight: 600;
  color: #2c3e50;
  margin-bottom: 0.25rem;
}

.intermediate-description {
  font-size: 0.85rem;
  color: #7f8c8d;
  margin-bottom: 0.5rem;
}

.intermediate-value {
  font-family: 'Courier New', monospace;
  font-size: 0.85rem;
  color: #34495e;
  background: #f8f9fa;
  padding: 0.5rem;
  border-radius: 4px;
  overflow-x: auto;
}

.latex-content {
  margin-top: 0.5rem;
  padding: 0.75rem;
  background: white;
  border-radius: 4px;
  overflow-x: auto;
}
</style>
```

---

## Phase 4: Create Composables (1 hour)

### Step 4.1: Graph Data Composable

`frontend/src/composables/useGraphData.js`:

```javascript
import { useGraphStore } from '../stores/graphStore'

export const useGraphData = () => {
  const store = useGraphStore()

  const loadGraphData = async () => {
    store.setLoading(true)
    store.setError(null)

    try {
      const response = await fetch('/api/graph')
      const data = await response.json()
      store.setGraphData(data)
    } catch (error) {
      console.error('Error loading graph data:', error)
      store.setError('Failed to load graph data')
    } finally {
      store.setLoading(false)
    }
  }

  const loadProcessDetails = async (processId) => {
    try {
      const response = await fetch(`/api/process/${encodeURIComponent(processId)}`)
      const data = await response.json()

      if (data.error) {
        console.error('API returned error:', data.error)
        store.setError(data.error)
        return null
      }

      return data
    } catch (error) {
      console.error('Error loading process details:', error)
      store.setError('Failed to load process details')
      return null
    }
  }

  const loadFlowDetails = async (source, target, material) => {
    try {
      const response = await fetch(
        `/api/flow/${encodeURIComponent(source)}/${encodeURIComponent(target)}/${encodeURIComponent(material)}`
      )
      const data = await response.json()

      if (data.error) {
        store.setError(data.error)
        return null
      }

      return data
    } catch (error) {
      console.error('Error loading flow details:', error)
      store.setError('Failed to load flow details')
      return null
    }
  }

  return {
    loadGraphData,
    loadProcessDetails,
    loadFlowDetails
  }
}
```

### Step 4.2: MathJax Composable

`frontend/src/composables/useMathJax.js`:

```javascript
import { onMounted, nextTick } from 'vue'

export const useMathJax = () => {
  const renderMath = async () => {
    await nextTick()

    if (window.MathJax && window.MathJax.typesetPromise) {
      try {
        await window.MathJax.typesetPromise()
      } catch (err) {
        console.error('MathJax error:', err)
      }
    }
  }

  onMounted(() => {
    renderMath()
  })

  return {
    renderMath
  }
}
```

---

## Phase 5: Testing and Deployment (1-2 hours)

### Step 5.1: Development Mode Testing

Run both servers concurrently:

**Terminal 1 - Flask Backend:**
```bash
# In project root
python examples/visualize_demo.py
```

**Terminal 2 - Vue Frontend:**
```bash
# In frontend directory
cd src/flowprog/visualization/frontend
npm run dev
```

Visit `http://localhost:3000` to test the Vue app with hot reloading.

### Step 5.2: Production Build

Build the Vue app for production:

```bash
cd src/flowprog/visualization/frontend
npm run build
```

This outputs to `src/flowprog/visualization/static/dist/`

### Step 5.3: Update Flask Server for Production

Edit `server.py`:

```python
def run(self, host='127.0.0.1', port=5000, debug=True, dev_mode=False):
    """
    Run the Flask development server.

    Args:
        host: Server host
        port: Server port
        debug: Enable Flask debug mode
        dev_mode: If True, serve Vue dev server; if False, serve built assets
    """
    self.app.config['ENV'] = 'development' if dev_mode else 'production'

    print(f"\n{'='*60}")
    print(f"Starting FlowProg Model Visualization Server")
    print(f"{'='*60}")

    if dev_mode:
        print(f"\n🔧 Development Mode")
        print(f"Vue Frontend: http://localhost:3000")
        print(f"Flask Backend: http://{host}:{port}")
        print(f"\nMake sure to run 'npm run dev' in the frontend directory!")
    else:
        print(f"\n📦 Production Mode")
        print(f"Open your browser to: http://{host}:{port}")

    print(f"\nProcesses: {len(self.model.processes)}")
    print(f"Objects: {len(self.model.objects)}")
    print(f"\nPress Ctrl+C to stop the server\n")

    self.app.run(host=host, port=port, debug=debug)
```

### Step 5.4: Test Production Build

```bash
# Build frontend
cd src/flowprog/visualization/frontend
npm run build

# Run Flask in production mode
cd ../../../../  # Back to project root
python -c "
from examples.visualize_demo import model, recipe_data
from flowprog.visualization import VisualizationServer

server = VisualizationServer(model, recipe_data)
server.run(dev_mode=False)
"
```

Visit `http://localhost:5000` to test production build.

---

## Phase 6: Migration Checklist

### Functional Parity Checklist

- [ ] Graph loads and displays all nodes and edges
- [ ] Process nodes are clickable
- [ ] Flow edges are clickable
- [ ] Process details panel opens with correct data
- [ ] Flow details panel opens with correct data
- [ ] Background click closes panel
- [ ] "Fit to Screen" button works
- [ ] "Reset Layout" button works
- [ ] All three expression modes display correctly
- [ ] LaTeX math renders correctly
- [ ] History list displays when available
- [ ] Intermediates list displays when available
- [ ] Missing coefficient warning displays when applicable
- [ ] X = Y merged display works for processes without stock
- [ ] Panel slide animation works smoothly
- [ ] Responsive to window resizing

### Code Quality Checklist

- [ ] No console errors
- [ ] Components are under 150 lines each
- [ ] Styles are scoped to components
- [ ] No global state except in store
- [ ] API calls are in composables
- [ ] Props are properly typed
- [ ] Events are properly emitted
- [ ] MathJax renders after DOM updates
- [ ] Code is formatted consistently
- [ ] Comments explain complex logic

---

## Phase 7: Optional Enhancements

### Enhancement 1: Error Boundary

Create `frontend/src/components/ErrorBoundary.vue`:

```vue
<template>
  <div v-if="error" class="error-boundary">
    <h2>Something went wrong</h2>
    <pre>{{ error }}</pre>
    <button @click="reset">Try Again</button>
  </div>
  <slot v-else />
</template>

<script setup>
import { ref, onErrorCaptured } from 'vue'

const error = ref(null)

onErrorCaptured((err) => {
  error.value = err.toString()
  return false
})

const reset = () => {
  error.value = null
}
</script>
```

### Enhancement 2: Loading States

Add loading indicators in `GraphView.vue`:

```vue
<div v-if="store.state.isLoading" class="loading">
  <div class="spinner"></div>
  <p>Loading graph data...</p>
</div>
```

### Enhancement 3: Unit Tests

Install Vitest:
```bash
npm install -D vitest @vue/test-utils
```

Create `frontend/src/components/__tests__/ExpressionBox.test.js`:

```javascript
import { mount } from '@vue/test-utils'
import { describe, it, expect } from 'vitest'
import ExpressionBox from '../expressions/ExpressionBox.vue'

describe('ExpressionBox', () => {
  it('renders label and expression', () => {
    const wrapper = mount(ExpressionBox, {
      props: {
        label: 'Test Expression',
        expression: 'x + y',
        latex: 'x + y'
      }
    })

    expect(wrapper.text()).toContain('Test Expression')
    expect(wrapper.text()).toContain('x + y')
  })

  it('shows warning when provided', () => {
    const wrapper = mount(ExpressionBox, {
      props: {
        label: 'Test',
        expression: 'x',
        latex: 'x',
        warning: {
          title: 'Warning',
          message: 'Something is wrong'
        }
      }
    })

    expect(wrapper.find('.warning-box').exists()).toBe(true)
  })
})
```

---

## Troubleshooting

### Common Issues

**Issue: Cytoscape not rendering**
- Make sure dagre is registered: `cytoscape.use(cytoscapeDagre)`
- Check that container ref is mounted before initializing
- Verify container has explicit dimensions

**Issue: MathJax not rendering**
- Ensure MathJax script is loaded in index.html
- Call `renderMath()` after DOM updates using `nextTick()`
- Check browser console for MathJax errors

**Issue: API calls failing in development**
- Verify Vite proxy is configured correctly
- Check Flask server is running on port 5000
- Use browser network tab to inspect requests

**Issue: Hot reload not working**
- Restart Vite dev server
- Check for syntax errors
- Clear browser cache

**Issue: Production build doesn't work**
- Verify build output path in `vite.config.js`
- Check Flask template references correct files
- Ensure Flask is not in debug mode for production

---

## Performance Considerations

1. **Lazy Loading**: Large graphs should use virtual scrolling
2. **Debouncing**: Debounce resize events
3. **Memoization**: Use `computed()` for derived state
4. **Code Splitting**: Split large components into async imports

---

## Next Steps After Migration

1. **Add TypeScript** for type safety
2. **Implement tests** for critical components
3. **Add Storybook** for component documentation
4. **Optimize bundle size** with tree shaking
5. **Add PWA support** for offline use
6. **Create JupyterLab widget** as mentioned in docs

---

## Summary

This migration plan transforms a 557-line monolithic JavaScript file into a modular, maintainable Vue.js application with:

- **8 reusable components** (avg. 50-100 lines each)
- **Component-scoped styling** (no CSS conflicts)
- **Centralized state management** (predictable data flow)
- **Composable business logic** (testable, reusable)
- **Development and production modes** (optimal DX)

The total migration should take **6-8 hours** for a developer with intermediate Vue knowledge, and results in a codebase that is significantly easier to maintain, test, and extend.
