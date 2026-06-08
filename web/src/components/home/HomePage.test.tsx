import { describe, it, expect } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import { MemoryRouter } from "react-router";
import { HomePage } from "./HomePage";

describe("HomePage", () => {
  it("renders title and mode toggle", () => {
    render(
      <MemoryRouter>
        <HomePage />
      </MemoryRouter>
    );
    expect(screen.getByText("按时间")).toBeInTheDocument();
    expect(screen.getByText("按家族")).toBeInTheDocument();
  });

  it("toggles to family view shows family ids", async () => {
    render(
      <MemoryRouter>
        <HomePage />
      </MemoryRouter>
    );
    fireEvent.click(screen.getByText("按家族"));
    expect(await screen.findByText("01-cnn")).toBeInTheDocument();
  });
});
