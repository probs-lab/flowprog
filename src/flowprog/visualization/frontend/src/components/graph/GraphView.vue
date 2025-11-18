<template>
  <div class="graph-container">
    <div ref="cyContainer" class="cy-container"></div>
    <ControlBar @fit="fitGraph" @reset="resetLayout" />
  </div>
</template>

<script setup>
import { ref, onMounted, watch } from 'vue'
import cytoscape from 'cytoscape'
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
