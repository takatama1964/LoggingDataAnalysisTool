using Avalonia.Controls;
using Avalonia.Interactivity;
using Avalonia.Platform.Storage;
using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;

namespace DataAnalysisTool;

public partial class MainWindow : Window
{
    private string? _csvPath;

    public MainWindow()
    {
        InitializeComponent();
        ApplyPlotFont();
        PlotView.Refresh();
    }

    private async void OpenCsv_Click(object? sender, RoutedEventArgs e)
    {
        var files = await StorageProvider.OpenFilePickerAsync(
            new FilePickerOpenOptions
            {
                Title = "CSVファイルを選択",
                AllowMultiple = false
            });

        if (files.Count == 0)
            return;

        _csvPath = files[0].Path.LocalPath;

        var firstLine = File.ReadLines(_csvPath).First();
        var headers = firstLine.Split(',').Select(x => x.Trim()).ToArray();

        XColumnComboBox.Items.Clear();
        YCandidateListBox.Items.Clear();

        FirstYSelectedListBox.Items.Clear();
        SecondYSelectedListBox.Items.Clear();

        foreach (var h in headers)
        {
            string name = h.Trim();

            XColumnComboBox.Items.Add(name);
            YCandidateListBox.Items.Add(name);
        }

        if (headers.Length > 0)
        {
            XColumnComboBox.SelectedIndex = 0;
        }
    }

    private void AddToFirstY_Click(object? sender, RoutedEventArgs e)
    {
        AddSelectedCandidatesToList(FirstYSelectedListBox);
    }

    private void AddToSecondY_Click(object? sender, RoutedEventArgs e)
    {
        AddSelectedCandidatesToList(SecondYSelectedListBox);
    }

    private void AddSelectedCandidatesToList(ListBox targetListBox)
    {
        if (YCandidateListBox.SelectedItems == null)
            return;

        foreach (var selected in YCandidateListBox.SelectedItems)
        {
            string name = selected?.ToString() ?? "";

            bool alreadyExists = false;

            foreach (var item in targetListBox.Items)
            {
                if ((item?.ToString() ?? "") == name)
                {
                    alreadyExists = true;
                    break;
                }
            }

            if (!alreadyExists)
            {
                targetListBox.Items.Add(name);
            }
        }
    }

    private void RemoveFirstY_Click(object? sender, RoutedEventArgs e)
    {
        RemoveSelectedItemsFromList(FirstYSelectedListBox);
    }

    private void RemoveSecondY_Click(object? sender, RoutedEventArgs e)
    {
        RemoveSelectedItemsFromList(SecondYSelectedListBox);
    }

    private void RemoveSelectedItemsFromList(ListBox listBox)
    {
        if (listBox.SelectedItems == null)
            return;

        var itemsToRemove = listBox.SelectedItems
            .Cast<object>()
            .ToList();

        foreach (var item in itemsToRemove)
        {
            listBox.Items.Remove(item);
        }
    }

    private void DrawGraph_Click(object? sender, RoutedEventArgs e)
    {
        if (_csvPath == null)
            return;

        if (XColumnComboBox.SelectedItem == null)
            return;

        bool hasFirst = FirstYSelectedListBox.Items.Count > 0;
        bool hasSecond = SecondYSelectedListBox.Items.Count > 0;

        if (!hasFirst && !hasSecond)
            return;

        var lines = File.ReadAllLines(_csvPath);

        if (lines.Length < 2)
            return;

        var headers = lines[0].Split(',').Select(x => x.Trim()).ToArray();

        string xName = XColumnComboBox.SelectedItem.ToString() ?? "";
        int xIndex = Array.IndexOf(headers, xName);

        if (xIndex < 0)
            return;

        bool isDateTimeAxis = XAxisModeComboBox.SelectedIndex == 1;

        PlotView.Plot.Clear();
        ApplyPlotFont();

        PlotView.Plot.Axes.Right.IsVisible = hasSecond;

        DrawSelectedSeries(
            lines,
            headers,
            xIndex,
            FirstYSelectedListBox.Items,
            useRightAxis: false,
            isDateTimeAxis: isDateTimeAxis);

        DrawSelectedSeries(
            lines,
            headers,
            xIndex,
            SecondYSelectedListBox.Items,
            useRightAxis: true,
            isDateTimeAxis: isDateTimeAxis);

        string graphTitle = GraphTitleTextBox.Text ?? "CSV Graph";

        string xAxisTitle = string.IsNullOrWhiteSpace(XAxisTitleTextBox.Text)
            ? xName
            : XAxisTitleTextBox.Text;

        string firstYAxisTitle = FirstYAxisTitleTextBox.Text ?? "第1Y軸";
        string secondYAxisTitle = SecondYAxisTitleTextBox.Text ?? "第2Y軸";

        if (isDateTimeAxis)
        {
            PlotView.Plot.Axes.DateTimeTicksBottom();
        }

        PlotView.Plot.Title(graphTitle);
        PlotView.Plot.Axes.Bottom.Label.Text = xAxisTitle;
        PlotView.Plot.Axes.Bottom.Label.IsVisible = true;
        PlotView.Plot.Axes.Left.Label.Text = firstYAxisTitle;

        if (hasSecond)
        {
            PlotView.Plot.Axes.Right.Label.Text = secondYAxisTitle;
        }

        if (LegendCheckBox.IsChecked == true)
        {
            PlotView.Plot.ShowLegend();
        }
        else
        {
            PlotView.Plot.HideLegend();
        }

        PlotView.Refresh();
    }

    private void DrawSelectedSeries(
        string[] lines,
        string[] headers,
        int xIndex,
        System.Collections.IList? selectedItems,
        bool useRightAxis,
        bool isDateTimeAxis)
    {
        if (selectedItems == null)
            return;

        foreach (var item in selectedItems)
        {
            string yName = item?.ToString() ?? "";
            int yIndex = Array.IndexOf(headers, yName);

            if (yIndex < 0)
                continue;

            var xs = new List<double>();
            var ys = new List<double>();

            foreach (var line in lines.Skip(1))
            {
                var cols = line.Split(',');

                if (cols.Length <= Math.Max(xIndex, yIndex))
                    continue;

                if (TryParseXValue(cols[xIndex], isDateTimeAxis, out double x) &&
                    double.TryParse(cols[yIndex], out double y))
                {
                    xs.Add(x);
                    ys.Add(y);
                }
            }

            var scatter = PlotView.Plot.Add.Scatter(xs.ToArray(), ys.ToArray());
            scatter.LegendText = useRightAxis ? $"{yName} [第2Y軸]" : $"{yName} [第1Y軸]";

            if (useRightAxis)
            {
                scatter.Axes.YAxis = PlotView.Plot.Axes.Right;
            }
            else
            {
                scatter.Axes.YAxis = PlotView.Plot.Axes.Left;
            }
        }
    }

    private bool TryParseXValue(string text, bool isDateTimeAxis, out double x)
    {
        text = text.Trim();

        if (!isDateTimeAxis)
        {
            return double.TryParse(text, out x);
        }

        // シリアル値：1 = 1日
        if (double.TryParse(text, out x))
        {
            return true;
        }

        string[] formats =
        {
            "yyyy/MM/dd HH:mm:ss",
            "yyyy/M/d H:mm:ss",
            "yyyy-MM-dd HH:mm:ss",
            "yyyy-M-d H:mm:ss",
            "yyyy/MM/dd HH:mm:ss.fff",
            "yyyy/M/d H:mm:ss.fff",
            "yyyy-MM-dd HH:mm:ss.fff",
            "yyyy-M-d H:mm:ss.fff"
        };

        if (DateTime.TryParseExact(
                text,
                formats,
                CultureInfo.InvariantCulture,
                DateTimeStyles.None,
                out DateTime dt))
        {
            x = dt.ToOADate();
            return true;
        }

        if (DateTime.TryParse(text, out dt))
        {
            x = dt.ToOADate();
            return true;
        }

        x = double.NaN;
        return false;
    }

    private void ResetAxis_Click(object? sender, RoutedEventArgs e)
    {
        PlotView.Plot.Axes.AutoScale();
        PlotView.Refresh();
    }

    private void ApplyPlotFont()
    {
        if (OperatingSystem.IsWindows())
        {
            PlotView.Plot.Font.Set("Meiryo");
        }
        else if (OperatingSystem.IsMacOS())
        {
            PlotView.Plot.Font.Set("Hiragino Sans");
        }
        else
        {
            PlotView.Plot.Font.Set("Noto Sans CJK JP");
        }
    }

    private void ApplyAxisRange_Click(object? sender, RoutedEventArgs e)
    {
        if (double.TryParse(XAxisMinTextBox.Text, out double xMin) &&
            double.TryParse(XAxisMaxTextBox.Text, out double xMax))
        {
            PlotView.Plot.Axes.SetLimitsX(xMin, xMax);
        }

        if (double.TryParse(FirstYAxisMinTextBox.Text, out double y1Min) &&
            double.TryParse(FirstYAxisMaxTextBox.Text, out double y1Max))
        {
            PlotView.Plot.Axes.SetLimitsY(y1Min, y1Max, PlotView.Plot.Axes.Left);
        }

        if (PlotView.Plot.Axes.Right.IsVisible &&
            double.TryParse(SecondYAxisMinTextBox.Text, out double y2Min) &&
            double.TryParse(SecondYAxisMaxTextBox.Text, out double y2Max))
        {
            PlotView.Plot.Axes.SetLimitsY(y2Min, y2Max, PlotView.Plot.Axes.Right);
        }

        PlotView.Refresh();
    }

    private async void SaveGraphImage_Click(object? sender, RoutedEventArgs e)
    {
        var file = await StorageProvider.SaveFilePickerAsync(
            new FilePickerSaveOptions
            {
                Title = "グラフ画像を保存",
                SuggestedFileName = "graph.png",
                DefaultExtension = "png"
            });

        if (file == null)
            return;

        string path = file.Path.LocalPath;

        ApplyPlotFont();

        int width = Math.Max(800, (int)PlotView.Bounds.Width);
        int height = Math.Max(600, (int)PlotView.Bounds.Height);

        PlotView.Plot.SavePng(path, width, height);
    }
}