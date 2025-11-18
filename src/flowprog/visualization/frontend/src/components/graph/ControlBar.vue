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
