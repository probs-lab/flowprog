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
