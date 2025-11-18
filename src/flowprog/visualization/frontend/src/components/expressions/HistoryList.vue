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
