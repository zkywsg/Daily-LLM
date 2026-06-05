import { prehistoryNodes, timelineNodes } from "./timeline";

test("timeline starts at 1989 LeNet and covers through 2025", () => {
  expect(timelineNodes.map((node) => node.year)).toEqual([
    "1989",
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

test("prehistory holds the 4 pre-deep-learning milestones", () => {
  expect(prehistoryNodes.map((n) => n.year)).toEqual([
    "1948",
    "1958",
    "1986",
    "1997",
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

test("major nodes carry prerequisites and tracks", () => {
  const transformer = timelineNodes.find((node) => node.year === "2017");

  expect(transformer?.tracks).toContainEqual({
    label: "Transformer 架构",
    path: "../tracks/language/transformer-architecture/",
  });
  expect(transformer?.prerequisites.length).toBeGreaterThan(0);
});
