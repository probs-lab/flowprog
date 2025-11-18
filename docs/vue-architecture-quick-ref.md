# Vue.js Architecture Quick Reference

Quick reference for the Vue.js visualization app architecture.

## Component Hierarchy

```
App.vue
├── GraphView.vue (557 lines → 120 lines)
│   └── ControlBar.vue (inline → 40 lines)
│
└── DetailsPanel.vue (200+ lines → 60 lines)
    ├── ProcessPanel.vue (150 lines)
    │   └── ExpressionModes.vue (80 lines)
    │       ├── ExpressionBox.vue × 3 (70 lines, reused)
    │       ├── HistoryList.vue (40 lines)
    │       └── IntermediatesList.vue (60 lines)
    │
    └── FlowPanel.vue (100 lines)
        └── ExpressionModes.vue (same as above)
```

## Data Flow

```
User Interaction
      ↓
GraphView emits event
      ↓
App.vue handles event
      ↓
useGraphData composable makes API call
      ↓
graphStore updates state
      ↓
DetailsPanel re-renders (reactive)
      ↓
useMathJax renders LaTeX
```

## File Size Comparison

| File | Before | After | Reduction |
|------|--------|-------|-----------|
| app.js | 557 lines | Removed | 100% |
| index.html | 340 lines (286 CSS) | 30 lines | 91% |
| **New files** | - | **~850 lines total** | - |
| **Avg component** | - | **~70 lines** | - |

**Net Result:** Same functionality, but split into 12 maintainable files instead of 1 monolithic file.

## State Management

```javascript
// stores/graphStore.js
{
  state: {
    graphData: null,           // Nodes & edges from API
    selectedElement: null,     // Currently selected node/edge
    panelData: null,          // Data for details panel
    isPanelOpen: false,       // Panel visibility
    isLoading: false,         // Loading state
    error: null               // Error messages
  },
  actions: {
    setGraphData(),
    selectElement(),
    setPanelData(),
    closePanel(),
    setLoading(),
    setError()
  }
}
```

## Key Design Patterns

### 1. Composition API (Script Setup)
```vue
<script setup>
import { ref, computed, onMounted } from 'vue'

// Reactive state
const count = ref(0)

// Computed values
const doubled = computed(() => count.value * 2)

// Lifecycle
onMounted(() => {
  console.log('Component mounted')
})
</script>
```

### 2. Props Down, Events Up
```vue
<!-- Parent -->
<ChildComponent
  :data="parentData"
  @custom-event="handleEvent"
/>

<!-- Child -->
<script setup>
defineProps({ data: Object })
defineEmits(['custom-event'])
</script>
```

### 3. Composables for Logic Reuse
```javascript
// composables/useGraphData.js
export const useGraphData = () => {
  const store = useGraphStore()

  const loadGraphData = async () => {
    // API logic here
  }

  return { loadGraphData }
}

// Used in components:
const { loadGraphData } = useGraphData()
```

### 4. Scoped Styling
```vue
<style scoped>
/* Only applies to this component */
.button {
  color: blue;
}
</style>
```

## API Integration

### Development Mode (Hot Reload)
```
User Browser → http://localhost:3000 (Vite)
                    ↓ (proxy /api/*)
               http://localhost:5000 (Flask)
```

### Production Mode
```
User Browser → http://localhost:5000 (Flask)
                    ↓ (serves static/dist/)
               Vue App (built assets)
```

## Component Responsibilities

| Component | Responsibility | Lines | Reusability |
|-----------|---------------|-------|-------------|
| `App.vue` | Layout & routing | 80 | ❌ Root component |
| `GraphView.vue` | Cytoscape integration | 120 | ⚠️ Graph-specific |
| `ControlBar.vue` | Graph controls | 40 | ✅ Reusable |
| `DetailsPanel.vue` | Panel container | 60 | ✅ Reusable |
| `ProcessPanel.vue` | Process details | 150 | ⚠️ Domain-specific |
| `FlowPanel.vue` | Flow details | 100 | ⚠️ Domain-specific |
| `ExpressionModes.vue` | Three eval modes | 80 | ✅ Reusable |
| `ExpressionBox.vue` | Single expression | 70 | ✅ **Highly reusable** |
| `HistoryList.vue` | History items | 40 | ✅ Reusable |
| `IntermediatesList.vue` | Intermediate vars | 60 | ✅ Reusable |

## Development Commands

```bash
# Initial setup
npm create vite@latest frontend -- --template vue
cd frontend
npm install
npm install cytoscape cytoscape-dagre dagre

# Development (hot reload)
npm run dev          # Frontend on :3000
python examples/visualize_demo.py  # Backend on :5000

# Production build
npm run build        # Outputs to ../static/dist/

# Testing (if added)
npm run test
```

## Migration Checklist (Quick)

- [ ] Phase 1: Setup (1h)
  - [ ] Create Vite project
  - [ ] Configure proxy
  - [ ] Update Flask template

- [ ] Phase 2: Core structure (1h)
  - [ ] Create directory structure
  - [ ] Setup store
  - [ ] Create composables

- [ ] Phase 3: Components (2-3h)
  - [ ] App.vue
  - [ ] GraphView.vue
  - [ ] Panels (Details, Process, Flow)
  - [ ] Expression components

- [ ] Phase 4: Testing (1h)
  - [ ] Test dev mode
  - [ ] Test production build
  - [ ] Verify all features

## Common Gotchas

1. **MathJax timing**: Always call `renderMath()` after `nextTick()`
2. **Cytoscape ref**: Initialize only after ref is mounted
3. **Proxy config**: Must restart Vite after changing proxy
4. **Scoped styles**: Use `:deep()` for child component styling
5. **Reactive unwrap**: Use `.value` for refs in `<script>`, not in `<template>`

## Performance Tips

```javascript
// ✅ Good: Computed for derived state
const doubled = computed(() => count.value * 2)

// ❌ Bad: Function in template (re-runs every render)
<div>{{ count * 2 }}</div>

// ✅ Good: v-if for conditional rendering
<div v-if="show">Heavy component</div>

// ⚠️ Caution: v-show still renders
<div v-show="show">Heavy component</div>
```

## Next Steps After Migration

1. ✅ **Works**: Vue app with all features
2. 🔄 **Improve**: Add TypeScript
3. 🧪 **Test**: Add unit tests with Vitest
4. 📚 **Document**: Add Storybook
5. 🚀 **Optimize**: Code splitting, lazy loading
6. 📦 **Deploy**: CI/CD pipeline
7. 🎨 **Enhance**: Custom themes, dark mode
8. 📊 **Extend**: JupyterLab widget

## Getting Help

- **Vue Docs**: https://vuejs.org/guide/
- **Vite Docs**: https://vitejs.dev/guide/
- **Cytoscape.js**: https://js.cytoscape.org/
- **Full Migration Plan**: See `vue-migration-plan.md`

## Quick Decision Matrix

**Use Vue if:**
- ✅ App will grow in complexity
- ✅ Need component reusability
- ✅ Want scoped styling
- ✅ Plan to add features
- ✅ Team knows/wants to learn Vue

**Stay vanilla if:**
- ❌ App is very simple (<100 lines)
- ❌ No plans to extend
- ❌ Build step is problematic
- ❌ Team doesn't want framework

**For this project: ✅ Vue is recommended** due to existing complexity (557 lines) and planned JupyterLab migration.
