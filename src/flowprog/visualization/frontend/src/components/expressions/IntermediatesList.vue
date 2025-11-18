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
