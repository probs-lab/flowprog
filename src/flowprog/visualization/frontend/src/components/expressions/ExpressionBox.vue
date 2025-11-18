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
