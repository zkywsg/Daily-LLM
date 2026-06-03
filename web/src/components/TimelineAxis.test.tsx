import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { timelineNodes } from "../data/timeline";
import { TimelineAxis } from "./TimelineAxis";

test("marks the active timeline node for assistive technology", () => {
  render(
    <TimelineAxis
      activeYear="2017"
      nodes={timelineNodes}
      onSelect={() => {}}
      onOpenPrehistory={() => {}}
    />,
  );

  expect(
    screen.getByRole("button", { name: /2017 Transformer/ }),
  ).toHaveAttribute("aria-current", "step");
});

test("calls onSelect when a node is clicked", async () => {
  const selected: string[] = [];

  render(
    <TimelineAxis
      activeYear="2012"
      nodes={timelineNodes}
      onSelect={(year) => selected.push(year)}
      onOpenPrehistory={() => {}}
    />,
  );

  await userEvent.click(screen.getByRole("button", { name: /2020 GPT-3/ }));

  expect(selected).toEqual(["2020"]);
});
