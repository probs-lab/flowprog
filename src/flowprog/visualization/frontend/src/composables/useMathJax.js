import { onMounted, nextTick } from 'vue'

export const useMathJax = () => {
  const renderMath = async () => {
    await nextTick()

    if (window.MathJax && window.MathJax.typesetPromise) {
      try {
        await window.MathJax.typesetPromise()
      } catch (err) {
        console.error('MathJax error:', err)
      }
    }
  }

  onMounted(() => {
    renderMath()
  })

  return {
    renderMath
  }
}
