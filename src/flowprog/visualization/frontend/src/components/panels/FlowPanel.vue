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
