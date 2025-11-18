import { reactive, readonly } from 'vue'

const state = reactive({
  graphData: null,
  selectedElement: null,
  panelData: null,
  isPanelOpen: false,
  isLoading: false,
  error: null
})

const actions = {
  setGraphData(data) {
    state.graphData = data
  },

  selectElement(element) {
    state.selectedElement = element
  },

  setPanelData(data) {
    state.panelData = data
    state.isPanelOpen = !!data
  },

  closePanel() {
    state.panelData = null
    state.isPanelOpen = false
    state.selectedElement = null
  },

  setLoading(loading) {
    state.isLoading = loading
  },

  setError(error) {
    state.error = error
  }
}

export const useGraphStore = () => {
  return {
    state: readonly(state),
    ...actions
  }
}
