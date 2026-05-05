# Timeline Visualization Page Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a deployable `web/` React application whose homepage is a warm, horizontal Daily-LLM timeline with clickable nodes and detailed content below.

**Architecture:** The app is a Vite + React + TypeScript single-page site. Timeline content lives in `web/src/data/timeline.ts`, interaction state lives in `web/src/App.tsx`, and focused components render the axis, content, work list, and module links.

**Tech Stack:** Vite, React, TypeScript, Vitest, React Testing Library, CSS.

---

## File Structure

- Create `web/package.json`: npm scripts and frontend dependencies.
- Create `web/index.html`: Vite mount point.
- Create `web/tsconfig.json`, `web/tsconfig.node.json`, `web/vite.config.ts`: TypeScript and test/build config.
- Create `web/src/main.tsx`: React entry point.
- Create `web/src/App.tsx`: selection state, hash sync, page shell.
- Create `web/src/data/timeline.ts`: `TimelineNode` type and timeline records.
- Create `web/src/components/TimelineAxis.tsx`: horizontal top timeline.
- Create `web/src/components/TimelineContent.tsx`: current node main reading content.
- Create `web/src/components/TimelineWorkList.tsx`: same-year key work list.
- Create `web/src/components/RelatedModules.tsx`: module links.
- Create `web/src/styles/global.css`: warm visual system, layout, responsive rules.
- Create `web/src/test/setup.ts`: DOM test setup.
- Create tests beside behavior: `timeline.test.ts`, `TimelineAxis.test.tsx`, `App.test.tsx`.

---

### Task 1: Scaffold Vite React Workspace

**Files:**
- Create: `web/package.json`
- Create: `web/index.html`
- Create: `web/tsconfig.json`
- Create: `web/tsconfig.node.json`
- Create: `web/vite.config.ts`
- Create: `web/src/test/setup.ts`

- [ ] **Step 1: Add frontend package and config files**

Create a Vite React package with scripts:

```json
{
  "name": "daily-llm-timeline",
  "private": true,
  "version": "0.1.0",
  "type": "module",
  "scripts": {
    "dev": "vite",
    "build": "tsc -b && vite build",
    "preview": "vite preview",
    "test": "vitest run"
  },
  "dependencies": {
    "@vitejs/plugin-react": "^5.0.0",
    "vite": "^7.0.0",
    "typescript": "^5.8.0",
    "react": "^19.0.0",
    "react-dom": "^19.0.0"
  },
  "devDependencies": {
    "@testing-library/jest-dom": "^6.6.0",
    "@testing-library/react": "^16.1.0",
    "@testing-library/user-event": "^14.5.0",
    "@types/react": "^19.0.0",
    "@types/react-dom": "^19.0.0",
    "jsdom": "^25.0.0",
    "vitest": "^3.0.0"
  }
}
```

- [ ] **Step 2: Install dependencies**

Run: `cd web && npm install`

Expected: `package-lock.json` is created and install completes without dependency errors.

- [ ] **Step 3: Verify empty toolchain**

Run: `cd web && npm test`

Expected: Vitest runs and reports no test files or passes after tests are added in later tasks.

---

### Task 2: Timeline Data Contract

**Files:**
- Create: `web/src/data/timeline.test.ts`
- Create: `web/src/data/timeline.ts`

- [ ] **Step 1: Write failing data tests**

Test that the data covers all required years, has usable section fields, and exposes module links.

```ts
import { timelineNodes } from "./timeline";

test("timeline covers the required historical nodes", () => {
  expect(timelineNodes.map((node) => node.year)).toEqual([
    "1948", "2012", "2013", "2014", "2015", "2016", "2017", "2018",
    "2019", "2020", "2021", "2022", "2023", "2024", "2025",
  ]);
});

test("each timeline node contains readable explanation sections", () => {
  for (const node of timelineNodes) {
    expect(node.title.length).toBeGreaterThan(4);
    expect(node.previousLimit.length).toBeGreaterThan(10);
    expect(node.whatHappened.length).toBeGreaterThan(10);
    expect(node.solved.length).toBeGreaterThan(10);
    expect(node.newProblems.length).toBeGreaterThan(10);
    expect(node.keyWorks.length).toBeGreaterThan(0);
  }
});

test("major nodes link back to existing learning modules", () => {
  const transformer = timelineNodes.find((node) => node.year === "2017");
  expect(transformer?.relatedModules).toContainEqual({
    label: "Transformer 架构",
    path: "../02-Language-Transformers/transformer-architecture/",
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd web && npm test -- src/data/timeline.test.ts`

Expected: FAIL because `./timeline` does not exist.

- [ ] **Step 3: Implement timeline data**

Define `TimelineNode`, `timelineNodes`, and `getNodeByYear`. Include 15 required nodes and enough Chinese content for the first page.

- [ ] **Step 4: Run test to verify it passes**

Run: `cd web && npm test -- src/data/timeline.test.ts`

Expected: PASS.

---

### Task 3: Interaction Shell and Hash Sync

**Files:**
- Create: `web/src/App.test.tsx`
- Create: `web/src/App.tsx`
- Create: `web/src/main.tsx`

- [ ] **Step 1: Write failing app tests**

Test default selection and node switching.

```tsx
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import App from "./App";

test("shows the 2012 node by default", () => {
  window.location.hash = "";
  render(<App />);
  expect(screen.getByRole("heading", { name: /AlexNet/ })).toBeInTheDocument();
});

test("clicking a timeline node updates the content and URL hash", async () => {
  window.location.hash = "";
  render(<App />);
  await userEvent.click(screen.getByRole("button", { name: /2017 Transformer/ }));
  expect(screen.getByRole("heading", { name: /Transformer/ })).toBeInTheDocument();
  expect(window.location.hash).toBe("#2017");
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd web && npm test -- src/App.test.tsx`

Expected: FAIL because `App` does not exist.

- [ ] **Step 3: Implement minimal app shell**

Render header, timeline buttons, and selected node content. Initialize from `window.location.hash` when it matches a node, otherwise default to `2012`.

- [ ] **Step 4: Run test to verify it passes**

Run: `cd web && npm test -- src/App.test.tsx`

Expected: PASS.

---

### Task 4: Focused Timeline Components

**Files:**
- Create: `web/src/components/TimelineAxis.test.tsx`
- Create: `web/src/components/TimelineAxis.tsx`
- Create: `web/src/components/TimelineContent.tsx`
- Create: `web/src/components/TimelineWorkList.tsx`
- Create: `web/src/components/RelatedModules.tsx`
- Modify: `web/src/App.tsx`

- [ ] **Step 1: Write failing axis tests**

```tsx
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { timelineNodes } from "../data/timeline";
import { TimelineAxis } from "./TimelineAxis";

test("marks the active timeline node for assistive technology", () => {
  render(<TimelineAxis nodes={timelineNodes} activeYear="2017" onSelect={() => {}} />);
  expect(screen.getByRole("button", { name: /2017 Transformer/ })).toHaveAttribute(
    "aria-current",
    "step",
  );
});

test("calls onSelect when a node is clicked", async () => {
  const selected: string[] = [];
  render(<TimelineAxis nodes={timelineNodes} activeYear="2012" onSelect={(year) => selected.push(year)} />);
  await userEvent.click(screen.getByRole("button", { name: /2020 GPT-3/ }));
  expect(selected).toEqual(["2020"]);
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd web && npm test -- src/components/TimelineAxis.test.tsx`

Expected: FAIL because component files do not exist.

- [ ] **Step 3: Extract focused components**

Move axis, content, work list, and related module rendering out of `App.tsx` into their own files. Keep component props typed and narrow.

- [ ] **Step 4: Run tests**

Run: `cd web && npm test`

Expected: PASS.

---

### Task 5: Warm Responsive Visual Layer

**Files:**
- Create: `web/src/styles/global.css`
- Modify: `web/src/main.tsx`
- Modify: component class names as needed.

- [ ] **Step 1: Add CSS for final layout**

Implement warm background, top horizontal axis, content grid, selected node state, focus states, and mobile single-column behavior.

- [ ] **Step 2: Run build**

Run: `cd web && npm run build`

Expected: TypeScript and Vite build pass.

- [ ] **Step 3: Run local dev server**

Run: `cd web && npm run dev -- --host 127.0.0.1`

Expected: Vite serves the app and prints a localhost URL.

---

### Task 6: Final Verification

**Files:**
- Modify only if verification finds issues.

- [ ] **Step 1: Run full tests**

Run: `cd web && npm test`

Expected: PASS.

- [ ] **Step 2: Run production build**

Run: `cd web && npm run build`

Expected: PASS.

- [ ] **Step 3: Inspect git diff**

Run: `git status --short` and `git diff -- web docs/superpowers/plans/2026-05-05-timeline-visualization-page.md`

Expected: Only planned files changed, aside from pre-existing unrelated workspace changes.
