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
