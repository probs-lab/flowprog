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
