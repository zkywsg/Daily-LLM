import { timelineNodes } from "./timeline";

test("timeline covers the required historical nodes", () => {
  expect(timelineNodes.map((node) => node.year)).toEqual([
    "1948",
    "2012",
    "2013",
    "2014",
    "2015",
    "2016",
    "2017",
    "2018",
    "2019",
    "2020",
    "2021",
    "2022",
    "2023",
    "2024",
    "2025",
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
