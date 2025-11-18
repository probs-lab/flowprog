import { useGraphStore } from '../stores/graphStore'

export const useGraphData = () => {
  const store = useGraphStore()

  const loadGraphData = async () => {
    store.setLoading(true)
    store.setError(null)

    try {
      const response = await fetch('/api/graph')
      const data = await response.json()
      store.setGraphData(data)
    } catch (error) {
      console.error('Error loading graph data:', error)
      store.setError('Failed to load graph data')
    } finally {
      store.setLoading(false)
    }
  }

  const loadProcessDetails = async (processId) => {
    try {
      const response = await fetch(`/api/process/${encodeURIComponent(processId)}`)
      const data = await response.json()

      if (data.error) {
        console.error('API returned error:', data.error)
        store.setError(data.error)
        return null
      }

      return data
    } catch (error) {
      console.error('Error loading process details:', error)
      store.setError('Failed to load process details')
      return null
    }
  }

  const loadFlowDetails = async (source, target, material) => {
    try {
      const response = await fetch(
        `/api/flow/${encodeURIComponent(source)}/${encodeURIComponent(target)}/${encodeURIComponent(material)}`
      )
      const data = await response.json()

      if (data.error) {
        store.setError(data.error)
        return null
      }

      return data
    } catch (error) {
      console.error('Error loading flow details:', error)
      store.setError('Failed to load flow details')
      return null
    }
  }

  return {
    loadGraphData,
    loadProcessDetails,
    loadFlowDetails
  }
}
