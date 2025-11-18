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
