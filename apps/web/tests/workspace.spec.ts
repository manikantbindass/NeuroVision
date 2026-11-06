import { expect, test, type Page } from "@playwright/test";

async function openPage(page: Page, label: string) {
  const navigation = page.getByRole("navigation", { name: "Research pages" });
  if (!(await navigation.isVisible()))
    await page.getByRole("button", { name: "Open navigation" }).click();
  await navigation.getByRole("link", { name: label, exact: true }).click();
}

test("guided demo links slice navigation, overlays, uncertainty and time points", async ({
  page,
}) => {
  const errors: string[] = [];
  page.on("pageerror", (error) => errors.push(error.message));
  await page.goto("/");
  await expect(
    page.getByRole("heading", { name: "A closer look at the human brain." }),
  ).toBeVisible();
  await page
    .locator(".hero")
    .getByRole("button", { name: "Start guided demo" })
    .click();
  await expect(
    page.getByRole("heading", { name: "MRI Workspace" }),
  ).toBeVisible();
  const axial = page.getByRole("slider", { name: "Axial slice" });
  await axial.focus();
  await axial.press("ArrowRight");
  await expect(
    page.getByRole("img", { name: "Axial MRI slice 34" }),
  ).toBeVisible();
  const sagittal = page.getByRole("slider", { name: "Sagittal slice" });
  await page
    .getByRole("img", { name: "Axial MRI slice 34" })
    .click({ position: { x: 9, y: 9 } });
  await expect(sagittal).not.toHaveValue("42");
  await page.getByRole("checkbox", { name: "Synthetic lesion" }).uncheck();
  await expect(
    page.getByRole("checkbox", { name: "Synthetic lesion" }),
  ).not.toBeChecked();

  await page.getByRole("button", { name: "Next" }).click();
  await expect(
    page.getByRole("heading", { name: "Segmentation Studio" }),
  ).toBeVisible();
  await page.getByLabel("Analytic mask").selectOption("1");
  await page.getByRole("button", { name: "3D", exact: true }).click();
  await expect(
    page.getByRole("img", {
      name: "Projected 3D segmentation boundary voxels",
    }),
  ).toBeVisible();
  await page.getByRole("slider", { name: "Rotation" }).focus();
  await page.getByRole("slider", { name: "Rotation" }).press("ArrowRight");

  await page.getByRole("button", { name: "Next" }).click();
  await expect(page.getByText("Uncalibrated model score")).toBeVisible();
  await page.getByRole("combobox", { name: "Overlay" }).selectOption("entropy");
  await expect(page.getByText("Brighter warm colors")).toBeVisible();

  await page.getByRole("button", { name: "Next" }).click();
  await expect(page.getByText("Observed differences may arise")).toBeVisible();
  await page.getByRole("button", { name: "Compare", exact: true }).click();
  await expect(page.getByText("After · day 90")).toBeVisible();
  await expect(
    page.getByRole("slider", { name: "Linked axial slice" }),
  ).toBeVisible();
  await page.getByRole("button", { name: "Finish" }).click();
  await expect(page.getByRole("region", { name: "Guided demo" })).toHaveCount(
    0,
  );
  expect(errors).toEqual([]);
});

test("mobile routes expose functional synthetic results and an accessible upload gate", async ({
  page,
}) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto("/");
  for (const [link, heading] of [
    ["Dataset Explorer", "NeuroVision synthetic phantom"],
    ["Preprocessing / QC", "Preprocessing & QC"],
    ["Model Comparison", "Prediction runs"],
    ["NeuroMap", "Regional overlap"],
    ["Metrics", "Overlap and distance metrics"],
    ["Reports", "AI-Assisted Research Segmentation Report"],
    ["MRI Upload", "Add a NIfTI volume"],
    ["System / Models", "Local API connection"],
  ]) {
    await openPage(page, link);
    await expect(page.getByRole("heading", { name: heading })).toBeVisible();
    expect(
      await page.evaluate(
        () => document.documentElement.scrollWidth <= window.innerWidth,
      ),
    ).toBeTruthy();
    if (link === "MRI Upload")
      await expect(
        page.getByRole("button", { name: "Upload volume" }),
      ).toBeDisabled();
  }
});

test("connected browser runs the synthetic research pipeline", async ({
  page,
}) => {
  const credential = process.env.NEUROVISION_WEB_TEST_TOKEN;
  test.skip(
    !credential,
    "Set a disposable local API token to run the connected workflow.",
  );
  await page.goto("/#system");
  await page.getByLabel("Bearer token").fill(credential!);
  await page.getByRole("button", { name: "Connect API", exact: true }).click();
  await expect(
    page.getByRole("button", { name: "API connected" }),
  ).toBeVisible();

  const nifti = Buffer.alloc(352 + 64 * 4);
  nifti.writeInt32LE(348, 0);
  [3, 4, 4, 4, 1, 1, 1, 1].forEach((value, i) =>
    nifti.writeInt16LE(value, 40 + i * 2),
  );
  nifti.writeInt16LE(16, 70);
  nifti.writeInt16LE(32, 72);
  [1, 1, 1, 1].forEach((value, i) => nifti.writeFloatLE(value, 76 + i * 4));
  nifti.writeFloatLE(352, 108);
  nifti.writeUInt8(2, 123);
  nifti.writeInt16LE(1, 254);
  [1, 1, 1].forEach((value, i) => nifti.writeFloatLE(value, 280 + i * 20));
  nifti.write("n+1\0", 344, "ascii");
  for (let i = 0; i < 64; i++) nifti.writeFloatLE(i / 63, 352 + 4 * i);

  await openPage(page, "MRI Upload");
  await page.getByLabel("NIfTI file").setInputFiles({
    name: "synthetic.nii",
    mimeType: "application/octet-stream",
    buffer: nifti,
  });
  await page.getByRole("checkbox").check();
  await page.getByRole("button", { name: "Upload volume" }).click();
  await expect(page.getByText("API research case")).toBeVisible();

  await openPage(page, "Preprocessing / QC");
  await page.getByLabel("Intensity recipe").selectOption("none");
  await page.getByRole("button", { name: "Queue preprocessing" }).click();
  await expect(page.getByText("QC status: pass")).toBeVisible();
  await page.getByRole("button", { name: "Compare", exact: true }).click();
  await expect(page.getByText("After · preprocessed")).toBeVisible();

  await openPage(page, "Segmentation Studio");
  await page.getByRole("checkbox", { name: "Use preprocessed volume" }).check();
  await page.getByRole("button", { name: "Queue inference" }).click();
  await expect(
    page.getByRole("checkbox", { name: "synthetic foreground" }),
  ).toBeChecked();
  await openPage(page, "NeuroConfidence");
  await expect(page.getByText("Uncalibrated model score")).toBeVisible();

  const reference = Buffer.from(nifti);
  for (let i = 0; i < 64; i++)
    reference.writeFloatLE(i >= 32 ? 1 : 0, 352 + 4 * i);
  await openPage(page, "MRI Upload");
  await page.getByLabel("NIfTI file").setInputFiles({
    name: "reference.nii",
    mimeType: "application/octet-stream",
    buffer: reference,
  });
  await page
    .getByRole("combobox", { name: "Input role" })
    .selectOption("labels");
  await page.getByRole("checkbox").check();
  await page.getByRole("button", { name: "Upload volume" }).click();
  await expect(page.getByText("Available as reference")).toBeVisible();

  await openPage(page, "Metrics");
  await page
    .getByRole("combobox", { name: "Target labels" })
    .selectOption({ index: 1 });
  await page.getByRole("button", { name: "Queue metrics" }).click();
  await expect(
    page.getByRole("cell", { name: "synthetic foreground" }),
  ).toBeVisible();
  await expect(page.getByRole("cell", { name: "1", exact: true })).toHaveCount(
    2,
  );

  await openPage(page, "NeuroMap");
  await page
    .getByRole("combobox", { name: "Atlas label map" })
    .selectOption({ index: 1 });
  await page.getByLabel("Atlas name").fill("Synthetic atlas");
  await page.getByLabel("Version").fill("1");
  await page.getByLabel("License").fill("MIT");
  await page.getByLabel("Reference space").fill("scanner");
  await page
    .getByLabel("Registration QC reference")
    .fill("Known shared fixture grid");
  await page
    .getByLabel("Atlas regions (JSON)")
    .fill('[{"code":1,"name":"Phantom region","hemisphere":"right"}]');
  await page
    .getByRole("checkbox", { name: "I reviewed the transform" })
    .check();
  await page.getByRole("button", { name: "Queue NeuroMap" }).click();
  await expect(
    page.getByRole("cell", { name: /Phantom region 100%/ }),
  ).toBeVisible();

  await openPage(page, "Reports");
  await page.getByRole("button", { name: "Queue report" }).click();
  await expect(
    page.getByRole("button", { name: "Download JSON" }),
  ).toBeVisible();
  const htmlDownload = page.waitForEvent("download");
  await page.getByRole("button", { name: "Download HTML" }).click();
  expect((await htmlDownload).suggestedFilename()).toBe("research-report.html");
  await expect(
    page.getByRole("button", { name: "Download PDF" }),
  ).toBeVisible();
  const pdfDownload = page.waitForEvent("download");
  await page.getByRole("button", { name: "Download PDF" }).click();
  expect((await pdfDownload).suggestedFilename()).toBe("research-report.pdf");
  await openPage(page, "System / Models");
  await page.getByRole("button", { name: "Disconnect API" }).click();
  await expect(page.getByRole("button", { name: "Local demo" })).toBeVisible();
  expect(
    await page.evaluate(() => localStorage.length + sessionStorage.length),
  ).toBe(0);
});
